import os
import cv2
import torch

from utils.utils import (
    LoadImages,
    driving_area_mask,
    lane_line_mask,
    non_max_suppression,
    plot_one_box,
    scale_coords,
    select_device,
    show_seg_result,
    split_for_trace_model,
    time_synchronized,
)


def run_yolopv2_inference(
    weights_path="data/weights/yolopv2.pt",
    image_path="data/example.jpg",
    device_id="0",
):


    device = select_device(device_id)

    print("=" * 60)
    print("YOLOPv2 Custom Inference")
    print("=" * 60)
    print(f"Device       : {device}")
    print(f"Weights      : {weights_path}")
    print(f"Input image  : {image_path}")


    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Model weights not found: {weights_path}"
        )

    print("\nLoading TorchScript model...")

    model = torch.jit.load(
        weights_path,
        map_location=device
    )

    model = model.to(device).float().eval()

    print("Model loaded successfully.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    print("\nLoading image...")

    dataset = LoadImages(
        image_path,
        img_size=640,
        stride=32
    )

    path, img, img0, vid_cap = next(iter(dataset))

    print(f"Original image shape : {img0.shape}")
    print(f"Preprocessed shape   : {img.shape}")

    img_tensor = torch.from_numpy(img).to(device).float()

    # Normalize [0, 255] -> [0, 1]
    img_tensor /= 255.0

    # Add batch dimension
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    print(f"Network input shape  : {img_tensor.shape}")


    print("\nWarming up model...")

    with torch.no_grad():
        _ = model(img_tensor)

    print("Warmup complete.")


    print("\nRunning inference...")

    with torch.no_grad():

        t1 = time_synchronized()

        [pred, anchor_grid], seg, ll = model(img_tensor)

        t2 = time_synchronized()

    inference_time = (t2 - t1) * 1000

    print(
        f"Inference latency   : {inference_time:.2f} ms"
    )
    print("\nRaw model outputs:")

    print(
        f"Detection prediction : "
        f"{pred.shape if hasattr(pred, 'shape') else type(pred)}"
    )

    print(
        f"Segmentation output  : {seg.shape}"
    )

    print(
        f"Lane output          : {ll.shape}"
    )

    print("\nProcessing object detections...")

    # Decode predictions produced by the traced model
    pred = split_for_trace_model(
        pred,
        anchor_grid
    )

    # Non-Maximum Suppression
    pred = non_max_suppression(
        pred,
        conf_thres=0.3,
        iou_thres=0.45
    )

    det = pred[0]

    if det is None:
        print("No objects detected.")
    else:
        print(
            f"Objects detected    : {len(det)}"
        )


    print("\nProcessing drivable-area segmentation...")

    da_seg_mask = driving_area_mask(seg)

    print(
        f"DA mask shape        : {da_seg_mask.shape}"
    )


    print("\nProcessing lane-line segmentation...")

    ll_seg_mask = lane_line_mask(ll)

    print(
        f"LL mask shape        : {ll_seg_mask.shape}"
    )


    print("\nMask information:")

    try:
        print(
            f"DA unique values     : "
            f"{torch.unique(da_seg_mask)}"
        )
    except Exception:
        print(
            f"DA unique values     : "
            f"{set(da_seg_mask.flatten())}"
        )

    try:
        print(
            f"LL unique values     : "
            f"{torch.unique(ll_seg_mask)}"
        )
    except Exception:
        print(
            f"LL unique values     : "
            f"{set(ll_seg_mask.flatten())}"
        )


    annotated = img0.copy()


    print("\nRendering segmentation...")

    show_seg_result(
        annotated,
        (
            da_seg_mask,
            ll_seg_mask
        ),
        is_demo=True
    )


    if det is not None and len(det):

        print("Rendering detections...")

        # Convert bounding boxes from network image coordinates
        # back to original image coordinates.
        det[:, :4] = scale_coords(
            img_tensor.shape[2:],
            det[:, :4],
            annotated.shape
        ).round()

        # Draw detections
        for *xyxy, conf, cls in reversed(det):

            label = f"Traffic {conf:.2f}"

            plot_one_box(
                xyxy,
                annotated,
                label=label,
                color=[0, 255, 0],
                line_thickness=2
            )


    out_path = "yolopv2_output.jpg"

    success = cv2.imwrite(
        out_path,
        annotated
    )

    if not success:
        raise RuntimeError(
            f"Failed to save output image: {out_path}"
        )

    print("\n" + "=" * 60)
    print("Inference complete")
    print("=" * 60)
    print(
        f"Output saved to: "
        f"{os.path.abspath(out_path)}"
    )
    print("=" * 60)


if __name__ == "__main__":

    run_yolopv2_inference(
        weights_path="data/weights/yolopv2.pt",
        image_path="data/example.jpg",
        device_id="0"
    )