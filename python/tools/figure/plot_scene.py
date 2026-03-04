# ========================================================================
# PLOT_SCENE: Visualize microphone array geometry and DOA estimation scenario
# ========================================================================
# This script creates a detailed engineering-style diagram showing:
# - A rectangular room with microphone array at the center
# - Speech source location and direction
# - Angular grid and azimuth plane visualization
# - End-fire regions (0-15°, 165-180°) for special focus
# - Sound wave propagation visualization
#
# Parameters:
#   - Room: 5m x 4m
#   - Microphone array: 4-element linear, 3.5cm spacing
#   - Angular range: 0-180° (azimuth plane, elevation=90°)
# ========================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_engineering_scenario():
    """
    Main function to draw the microphone array scenario diagram.
    
    Generates a publication-ready figure showing:
    - Room boundaries and dimensions
    - Coordinate system with polar grid
    - Microphone array with spacing annotation
    - Sound source and propagation visualization
    - End-fire region highlights
    """
    # --- 1. Canvas setup ---
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 30
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'

    # Make the canvas slightly taller to leave space for top annotations
    fig, ax = plt.subplots(figsize=(10, 9.5))

    # --- 2. Geometry parameters (1 unit = 1 meter) ---
    room_w = 5.0
    room_h = 4.0

    # Coordinate system: bottom center (0,0) -> x:[-2.5, 2.5], y:[0, 4]

    # Source parameters
    eng_angle = 127
    math_angle = 180 - eng_angle
    dist_r = 3.2
    rad = np.radians(math_angle)
    src_x = dist_r * np.cos(rad)
    src_y = dist_r * np.sin(rad)

    # --- 3. Draw the room ---
    # Create rectangular room boundary with light gray background
    room_rect = patches.Rectangle((-room_w / 2, 0), room_w, room_h,
                                  linewidth=3, edgecolor='#444444', facecolor='#F9F9F9', zorder=0)
    ax.add_patch(room_rect)

    # --- 4. Helper: engineering-style dimension annotation ---
    def draw_eng_dim(p1, p2, text, offset_dist, text_offset=0.2, color='black'):
        """
        Draw engineering-style dimension annotations.
        
        Args:
            p1, p2: (x, y) tuples for start/end points of the measured segment
            text: Dimension label text
            offset_dist: Perpendicular offset distance from the measured points
                        (+ up/right, - down/left)
            text_offset: Additional text offset for label placement
            color: Color of dimension lines and text
        """
        x1, y1 = p1
        x2, y2 = p2

        # Determine horizontal vs vertical dimension
        if abs(x1 - x2) > abs(y1 - y2):  # horizontal dimension
            dim_y = y1 + offset_dist
            # 1) Extension lines
            # Add a small gap so lines don't touch the object directly
            gap = 0.1 * (1 if offset_dist > 0 else -1)
            overhang = 0.2 * (1 if offset_dist > 0 else -1)

            ax.plot([x1, x1], [y1 + gap, dim_y + overhang], color=color, lw=0.8)
            ax.plot([x2, x2], [y2 + gap, dim_y + overhang], color=color, lw=0.8)

            # 2) Dimension line
            ax.annotate("", xy=(x1, dim_y), xytext=(x2, dim_y),
                        arrowprops=dict(arrowstyle='<|-|>', color=color, lw=1.2))

            # 3) Text label
            ax.text((x1 + x2) / 2, dim_y + text_offset * (1 if offset_dist > 0 else -1.5), text,
                    ha='center', va='bottom' if offset_dist > 0 else 'top',
                    fontsize=30, fontweight='bold', color=color)

        else:  # vertical dimension
            dim_x = x1 + offset_dist
            gap = 0.1 * (1 if offset_dist > 0 else -1)
            overhang = 0.2 * (1 if offset_dist > 0 else -1)

            ax.plot([x1 + gap, dim_x + overhang], [y1, y1], color=color, lw=0.8)
            ax.plot([x2 + gap, dim_x + overhang], [y2, y2], color=color, lw=0.8)

            ax.annotate("", xy=(dim_x, y1), xytext=(dim_x, y2),
                        arrowprops=dict(arrowstyle='<|-|>', color=color, lw=1.2))

            ax.text(dim_x + text_offset * (1 if offset_dist > 0 else -1.2), (y1 + y2) / 2, text,
                    ha='left' if offset_dist > 0 else 'right', va='center', rotation=90,
                    fontsize=30, fontweight='bold', color=color)

    # --- 5. Draw polar grid ---
    # Angular grid lines from center to room boundaries
    def get_intersection(angle_deg):
        """
        Calculate intersection point of an angle with room boundary.
        
        Args:
            angle_deg: Angle in degrees from horizontal
        
        Returns:
            Tuple[float, float]: (x, y) coordinates of intersection point
        """
        r_rad = np.radians(angle_deg)
        dx, dy = np.cos(r_rad), np.sin(r_rad)
        if dy == 0:
            return (room_w / 2 if dx > 0 else -room_w / 2), 0
        if dx == 0:
            return 0, room_h
        t = min(t for t in [room_h / dy, (room_w / 2) / dx if dx > 0 else (-room_w / 2) / dx] if t > 0.01)
        return t * dx, t * dy

    grid_angles = range(0, 181, 30)
    for math_ang in grid_angles:
        end_x, end_y = get_intersection(math_ang)

        ax.plot([0, end_x], [0, end_y], color='#AAAAAA', linestyle=':', linewidth=1.2, zorder=1)

        # Adjust label placement
        txt_x, txt_y = end_x * 0.90, end_y * 0.90
        ha, va = 'center', 'center'

        if math_ang == 0:
            txt_x = room_w / 2 - 0.1
            txt_y = 0.20
            ha = 'right'
            va = 'bottom'
        elif math_ang == 180:
            txt_x = -room_w / 2 + 0.1
            txt_y = 0.20
            ha = 'left'
            va = 'bottom'
        elif math_ang == 90:
            # Place 90° inside the room at the top edge to avoid the top dimension line
            txt_x = 0
            txt_y = room_h - 0.2
            va = 'top'
        elif math_ang == 150:
            txt_x += 0.15
        elif math_ang == 30:
            txt_x -= 0.10

        ax.text(txt_x, txt_y, r"${}^\circ$".format(math_ang),
                ha=ha, va=va, fontsize=25, color='#333333', fontweight='bold')

    # --- 6. End-fire zones ---
    # Highlight critical end-fire regions (angles 0-15° and 165-180°)
    zone_r = 1.6
    ax.add_patch(patches.Wedge((0, 0), zone_r, 165, 180, color='#E74C3C', alpha=0.15))  # left
    ax.text(-1.3, -0.35, "End-fire", color='#922B21', fontsize=30, ha='center', fontweight='bold')

    ax.add_patch(patches.Wedge((0, 0), zone_r, 0, 15, color='#E74C3C', alpha=0.15))  # right
    ax.text(1.3, -0.35, "End-fire", color='#922B21', fontsize=30, ha='center', fontweight='bold')

    # --- 7. Source and wavefronts ---
    # Draw source position, acoustic rays, and propagating wavefronts
    ax.plot([0, src_x], [0, src_y], color='#2980B9', linestyle='-.', linewidth=1.5, zorder=2)
    r_label_x = src_x * 0.52 - 0.10
    r_label_y = src_y * 0.52 + 0.20
    ax.text(r_label_x, r_label_y, r"$R=1\,\mathrm{m}/2\,\mathrm{m}$",
            color='#2980B9', fontsize=24, fontweight='bold', ha='right')
    ax.plot([src_x * 0.50, r_label_x - 0.40], [src_y * 0.50, r_label_y - 0.10],
            color='#2980B9', linewidth=1.2, zorder=3)

    # Angle annotation between 0° and the source direction (theta)
    theta_arc_r = 0.55
    theta_arc = patches.Arc((0, 0), 2 * theta_arc_r, 2 * theta_arc_r,
                            angle=0.0, theta1=0, theta2=math_angle,
                            color='#2980B9', linewidth=2.2, zorder=6)
    ax.add_patch(theta_arc)

    theta_mid = np.radians(math_angle / 2)
    theta_txt_r = theta_arc_r + 0.12
    ax.text(theta_txt_r * np.cos(theta_mid), theta_txt_r * np.sin(theta_mid), r"$\theta$",
            color='#2980B9', fontsize=26, fontweight='bold', ha='center', va='center', zorder=7)

    ax.scatter(src_x, src_y, color='#E74C3C', s=300, zorder=10, edgecolor='white', linewidth=2)
    ax.text(src_x - 1.0, src_y - 0.2, "Speech\nSource", ha='center', va='bottom', fontsize=30, fontweight='bold')

    # Wavefront arcs (near the source)
    prop_dir = math_angle + 180
    for r in np.linspace(0.3, 0.7, 6):
        arc = patches.Arc((src_x, src_y), 2 * r, 2 * r, angle=0.0,
                          theta1=prop_dir - 10, theta2=prop_dir + 10,
                          color='#C0392B', linewidth=2.5, alpha=0.7)
        ax.add_patch(arc)

    # --- 8. Microphone array (d annotation) ---
    # Draw 4-element linear microphone array at the origin with spacing (d) annotation
    vis_width = 0.8
    mic_x = np.linspace(-vis_width / 2, vis_width / 2, 4)
    ax.plot(mic_x, np.zeros_like(mic_x), color='black', linewidth=3, zorder=10)
    ax.scatter(mic_x, np.zeros_like(mic_x), color='black', s=120, zorder=11)
    # d annotation (keep only one) + clear extension lines
    dim_y = -0.25
    gap = -0.02
    overhang = -0.06
    ax.plot([mic_x[1], mic_x[1]], [gap, dim_y + overhang], color='black', lw=0.8)
    ax.plot([mic_x[2], mic_x[2]], [gap, dim_y + overhang], color='black', lw=0.8)
    ax.annotate("", xy=(mic_x[1], dim_y), xytext=(mic_x[2], dim_y),
                arrowprops=dict(arrowstyle='<|-|>', color='black', lw=1.0, mutation_scale=10))
    ax.text((mic_x[1] + mic_x[2]) / 2, dim_y - 0.20, r"$d=3.5\,\mathrm{cm}$",
            ha='center', va='top', fontsize=30, color='black', fontweight='bold')

    # --- 9. Room dimension annotations (using the helper function) ---
    # Add engineering-style dimension lines and labels
    # Top 5 m (offset upward by 0.7)
    draw_eng_dim((-2.5, 3.5), (2.5, 3.5), "5 m", offset_dist=0.7)

    # Left side 4 m (offset left by 0.7)
    draw_eng_dim((-2, 0), (-2, 4), "4 m", offset_dist=-0.7)

    # --- 11. Styling / finishing touches ---
    # Set aspect ratio, hide axes, and configure view limits for final presentation
    ax.set_aspect('equal')
    ax.axis('off')

    # View limits
    plt.xlim(-4.2, 4.2)
    plt.ylim(-1.5, 5.5)  # Leave top space for the 5 m annotation

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    draw_engineering_scenario()
