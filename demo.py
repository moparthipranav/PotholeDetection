import argparse
import json
import math
import time
from pathlib import Path
import cv2
import torch

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path,\
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model,\
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result,\
    AverageMeter,\
    LoadImages

from egoroad_pipeline import CameraConfig, EgoRoadPipeline
from yolov8_adapter import ultralytics_result_to_detections

TRAFFIC_CLASS_NAMES = [
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=['data/weights/yolopv2.pt'], help='model.pt path(s)')
    parser.add_argument('--pothole-weights', type=str, default='best.pt', help='YOLOv8 pothole model path; pass an empty string to disable')
    parser.add_argument('--pothole-conf-thres', type=float, default=0.25, help='pothole confidence threshold')
    parser.add_argument('--pothole-class', type=int, default=5, help='pothole class id in the YOLOv8 model')
    parser.add_argument('--pothole-img-size', type=int, default=640, help='YOLOv8 inference size (pixels)')
    parser.add_argument('--camera-fx', type=float, default=900.0, help='camera focal length x in pixels after resize')
    parser.add_argument('--camera-fy', type=float, default=900.0, help='camera focal length y in pixels after resize')
    parser.add_argument('--camera-cx', type=float, default=640.0, help='camera principal point x after resize')
    parser.add_argument('--camera-cy', type=float, default=360.0, help='camera principal point y after resize')
    parser.add_argument('--camera-height', type=float, default=1.5, help='camera height above road in meters')
    parser.add_argument('--camera-pitch', type=float, default=2.0, help='camera pitch in degrees')
    parser.add_argument('--no-sas-json', action='store_true', help='disable per-frame situational awareness JSONL output')
    parser.add_argument('--source', type=str, default='data/example.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser


def draw_situational_awareness(image, features, decision):
    width = image.shape[1]
    panel_width = min(720, width - 20)
    panel_height = 122
    level_color = {
        'HIGH': (50, 170, 50),
        'MODERATE': (0, 180, 230),
        'LOW': (0, 0, 220),
    }.get(decision.awareness, (90, 90, 90))

    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (20, 24, 24), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)
    cv2.rectangle(image, (10, 10), (10 + panel_width, 10 + panel_height), level_color, 2)

    nearest_distance = (
        f'{features.nearest_object_distance_m:.1f} m'
        if math.isfinite(features.nearest_object_distance_m) else 'n/a'
    )
    ttc = f'{features.ttc_s:.1f} s' if math.isfinite(features.ttc_s) else 'n/a'
    pothole_distance = (
        f'{features.nearest_pothole_distance_m:.1f} m'
        if math.isfinite(features.nearest_pothole_distance_m) else 'n/a'
    )
    awareness = decision.awareness if features.feature_valid else f'{decision.awareness} - LIMITED DATA'
    lines = [
        (f'Situational awareness  {decision.sas:.1f}/10  {awareness}', level_color, 0.62, 2),
        (f'Lane deviation {features.lane_deviation_m:.2f} m   Drivable {features.drivable_ratio:.0%}', (245, 245, 245), 0.48, 1),
        (f'Nearest {features.nearest_object_class}: {nearest_distance}   TTC {ttc}', (245, 245, 245), 0.48, 1),
        (f'Pothole severity {features.pothole_severity:.2f}   {pothole_distance}   Brake {decision.braking_intensity:.0%} {decision.steering}', (245, 245, 245), 0.43, 1),
    ]
    for index, (text, color, scale, thickness) in enumerate(lines):
        cv2.putText(image, text, (22, 35 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def draw_pothole_detections(image, result, class_names):
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].round().int().tolist()
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        label = f'{class_names[class_id]} {confidence:.2f}'
        color = (0, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        label_top = max(y1 - text_height - baseline - 6, 0)
        cv2.rectangle(
            image,
            (x1, label_top),
            (x1 + text_width + 8, label_top + text_height + baseline + 6),
            color,
            -1,
        )
        cv2.putText(
            image,
            label,
            (x1 + 4, label_top + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def detect():
    # setting and directories
    source, weights,  save_txt, imgsz = opt.source, opt.weights,  opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride =32
    model  = torch.jit.load(weights[0])
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16  
    model.eval()

    awareness_pipeline = EgoRoadPipeline(CameraConfig(
        fx=opt.camera_fx,
        fy=opt.camera_fy,
        cx=opt.camera_cx,
        cy=opt.camera_cy,
        height_m=opt.camera_height,
        pitch_deg=opt.camera_pitch,
    ))

    pothole_model = None
    if opt.pothole_weights:
        pothole_weights = Path(opt.pothole_weights)
        if not pothole_weights.is_file():
            raise FileNotFoundError(f'YOLOv8 pothole weights not found: {pothole_weights}')
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError('Install the YOLOv8 dependency with: pip install ultralytics') from exc
        pothole_model = YOLO(str(pothole_weights))

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred,anchor_grid],seg,ll= model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of  torch.jit.trace causes extra time consumption in demo version 
        # but this problem will not appear in offical version 
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred,anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)
        pothole_result = None
        if pothole_model is not None:
            pothole_result = pothole_model.predict(
                source=im0s,
                imgsz=opt.pothole_img_size,
                conf=opt.pothole_conf_thres,
                classes=[opt.pothole_class],
                device=0 if device.type == 'cuda' else 'cpu',
                verbose=False,
            )[0]

        # Process detections
        for i, det in enumerate(pred):  # detections per image
          
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            traffic_detections = []
            for box in det:
                x1, y1, x2, y2, confidence, class_id = box.tolist()
                class_id = int(class_id)
                class_name = (
                    TRAFFIC_CLASS_NAMES[class_id]
                    if 0 <= class_id < len(TRAFFIC_CLASS_NAMES)
                    else f'traffic_object_{class_id}'
                )
                traffic_detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'conf': confidence,
                    'class_id': class_id,
                    'class_name': class_name,
                })

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    #s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img :  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            show_seg_result(im0, (da_seg_mask,ll_seg_mask), is_demo=True)
            if pothole_result is not None:
                draw_pothole_detections(im0, pothole_result, pothole_model.names)

            pothole_detections = ultralytics_result_to_detections(
                pothole_result,
                pothole_model.names if pothole_model is not None else None,
            )
            features, decision = awareness_pipeline.process(
                {
                    'lane_mask': ll_seg_mask,
                    'drive_mask': da_seg_mask,
                    'traffic': traffic_detections,
                },
                pothole_detections,
            )
            draw_situational_awareness(im0, features, decision)

            ttc_text = f'{features.ttc_s:.1f}s' if math.isfinite(features.ttc_s) else 'n/a'
            data_status = '' if features.feature_valid else ' (LIMITED DATA)'
            print(
                f'SAS {decision.sas:.1f}/10 {decision.awareness}{data_status}; '
                f'TTC {ttc_text}; pothole severity {features.pothole_severity:.2f}'
            )
            if not opt.no_sas_json:
                record = {
                    'source': str(p),
                    'frame': frame,
                    **EgoRoadPipeline.to_dict(features, decision),
                }
                with open(save_dir / 'situational_awareness.jsonl', 'a', encoding='utf-8') as output_file:
                    output_file.write(json.dumps(_json_safe(record), allow_nan=False) + '\n')

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            #w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            #h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            w,h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    inf_time.update(t2-t1,img.size(0))
    nms_time.update(t4-t3,img.size(0))
    waste_time.update(tw2-tw1,img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    opt =  make_parser().parse_args()
    print(opt)

    with torch.no_grad():
            detect()
