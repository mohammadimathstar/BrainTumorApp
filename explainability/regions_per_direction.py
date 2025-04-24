
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def k_largest_index_argsort(a, k):

    idx = np.argsort(a.ravel())[:-k-1:-1]
    return np.column_stack(np.unravel_index(idx, a.shape))


def plot_important_region_per_principal_direction(
        img,
        region_map,
        result_dir):

    imgW, imgH = 224, 224
    regionW, regionH = region_map[0].shape[-2:]
    region_size = (int(imgW / regionW), int(imgH / regionH))

    # Predefined set of distinguishable colors (BGR)
    color_palette = [
        (255, 0, 0),    # 0 - Blue
        (0, 255, 0),    # 1 - Green
        (0, 0, 255),    # 2 - Red
        (255, 255, 0),  # 3 - Cyan
        (255, 0, 255),  # 4 - Magenta
        (0, 255, 255),  # 5 - Yellow
        (128, 0, 128),  # 6 - Purple
        (0, 128, 255),  # 7 - Orange-ish
        (0, 0, 128),    # 8 - Maroon
        (128, 128, 0),  # 9 - Olive
    ]

    # Track which regions were already picked and by which directions
    selected_regions = {}
    for direction_idx, region_per_dir in enumerate(region_map):
        region_per_dir = region_per_dir.numpy()
        ids = k_largest_index_argsort(region_per_dir, k=1)
        color = color_palette[direction_idx % len(color_palette)]

        for row_idx, col_idx in ids:
            key = (row_idx, col_idx)
            selected_regions.setdefault(key, []).append(direction_idx)

            offset = 3 * (len(selected_regions[key]) - 1)  # Offset for overlapping rectangles

            start_point = (
                col_idx * region_size[0] + offset+1,
                row_idx * region_size[1] + offset+1
            )
            end_point = (
                (col_idx + 1) * region_size[0] - offset-1,
                (row_idx + 1) * region_size[1] - offset-1
            )

            image = cv2.rectangle(img, start_point, end_point, color, thickness=2)

    # Draw legend (bottom-left corner)
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 0.9
    font_thickness = 1

    legend_x= 10 # legend near left edge
    legend_x = 224 - 40  # Push legend near right edge
    legend_y = 20
    line_spacing = 15

    for i in range(len(region_map)):
        color = color_palette[i % len(color_palette)]
        label = f"{i+1}"
        cv2.putText(image, label, (legend_x + 20, legend_y+4 + i * line_spacing),
                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        cv2.rectangle(image, (legend_x, legend_y - 7 + i * line_spacing),
                      (legend_x + 15, legend_y + 5 + i * line_spacing),
                      color, -1)

    # final_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imsave(
        fname=os.path.join(result_dir, 'highlighted_regions.png'),
        arr=image,
        vmin=0.0, vmax=1.0
    )

