import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_engineering_scenario():
    # --- 1. 画布设置 ---
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 30
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    
    # 画布稍微高一点，留出顶部标注空间
    fig, ax = plt.subplots(figsize=(10, 9.5))
    
    # --- 2. 几何参数 (1 unit = 1 meter) ---
    room_w = 5.0
    room_h = 4.0
    
    # 坐标系: 底部中心 (0,0) -> x:[-2.5, 2.5], y:[0, 4]
    
    # 声源参数
    eng_angle = 127
    math_angle = 180 - eng_angle
    dist_r = 3.2 
    rad = np.radians(math_angle)
    src_x = dist_r * np.cos(rad)
    src_y = dist_r * np.sin(rad)

    # --- 3. 绘制房间 ---
    room_rect = patches.Rectangle((-room_w/2, 0), room_w, room_h, 
                                  linewidth=3, edgecolor='#444444', facecolor='#F9F9F9', zorder=0)
    ax.add_patch(room_rect)

    # --- 4. 辅助绘图函数: 工程尺寸标注 ---
    def draw_eng_dim(p1, p2, text, offset_dist, text_offset=0.2, color='black'):
        """
        绘制工程风格的尺寸标注
        p1, p2: (x, y) 元组，测量起始点和终止点
        offset_dist: 尺寸线距离测量点的垂直距离 (+向上/右, -向下/左)
        """
        x1, y1 = p1
        x2, y2 = p2
        
        # 判断是水平还是垂直标注
        if abs(x1 - x2) > abs(y1 - y2): # 水平标注
            dim_y = y1 + offset_dist
            # 1. 引出线 (Extension lines)
            # 稍微留一点空隙 gap，不要紧贴物体
            gap = 0.1 * (1 if offset_dist > 0 else -1)
            overhang = 0.2 * (1 if offset_dist > 0 else -1)
            
            ax.plot([x1, x1], [y1 + gap, dim_y + overhang], color=color, lw=0.8)
            ax.plot([x2, x2], [y2 + gap, dim_y + overhang], color=color, lw=0.8)
            
            # 2. 尺寸线 (Dimension line)
            ax.annotate("", xy=(x1, dim_y), xytext=(x2, dim_y),
                        arrowprops=dict(arrowstyle='<|-|>', color=color, lw=1.2))
            
            # 3. 文字
            ax.text((x1+x2)/2, dim_y + text_offset * (1 if offset_dist > 0 else -1.5), text, 
                    ha='center', va='bottom' if offset_dist > 0 else 'top', 
                    fontsize=30, fontweight='bold', color=color)
            
        else: # 垂直标注
            dim_x = x1 + offset_dist
            gap = 0.1 * (1 if offset_dist > 0 else -1)
            overhang = 0.2 * (1 if offset_dist > 0 else -1)
            
            ax.plot([x1 + gap, dim_x + overhang], [y1, y1], color=color, lw=0.8)
            ax.plot([x2 + gap, dim_x + overhang], [y2, y2], color=color, lw=0.8)
            
            ax.annotate("", xy=(dim_x, y1), xytext=(dim_x, y2),
                        arrowprops=dict(arrowstyle='<|-|>', color=color, lw=1.2))
            
            ax.text(dim_x + text_offset * (1 if offset_dist > 0 else -1.2), (y1+y2)/2, text, 
                    ha='left' if offset_dist > 0 else 'right', va='center', rotation=90,
                    fontsize=30, fontweight='bold', color=color)

    # --- 5. 绘制极坐标网格 ---
    def get_intersection(angle_deg):
        r_rad = np.radians(angle_deg)
        dx, dy = np.cos(r_rad), np.sin(r_rad)
        if dy == 0: return (room_w/2 if dx > 0 else -room_w/2), 0
        if dx == 0: return 0, room_h
        t = min(t for t in [room_h/dy, (room_w/2)/dx if dx>0 else (-room_w/2)/dx] if t > 0.01)
        return t*dx, t*dy

    grid_angles = range(0, 181, 30)
    for math_ang in grid_angles:
        end_x, end_y = get_intersection(math_ang)
        
        ax.plot([0, end_x], [0, end_y], color='#AAAAAA', linestyle=':', linewidth=1.2, zorder=1)
        
        # 标注文字调整
        txt_x, txt_y = end_x * 0.90, end_y * 0.90
        ha, va = 'center', 'center'
        
        if math_ang == 0: 
            txt_x = room_w/2 - 0.1; txt_y = 0.20; ha = 'right'; va = 'bottom'
        elif math_ang == 180: 
            txt_x = -room_w/2 + 0.1; txt_y = 0.20; ha = 'left'; va = 'bottom'
        elif math_ang == 90: 
            # 修改：90度放在房间内部顶端，避开上方尺寸线
            txt_x = 0; txt_y = room_h - 0.2
            va = 'top'
        elif math_ang == 150:
            txt_x += 0.15
        elif math_ang == 30:
            txt_x -= 0.10
        
        ax.text(txt_x, txt_y, r"${}^\circ$".format(math_ang), 
            ha=ha, va=va, fontsize=25, color='#333333', fontweight='bold')

    # --- 6. End-fire Zones ---
    zone_r = 1.6
    ax.add_patch(patches.Wedge((0,0), zone_r, 165, 180, color='#E74C3C', alpha=0.15)) # 左
    ax.text(-1.3, -0.35, "End-fire", color='#922B21', fontsize=30, ha='center', fontweight='bold')
    
    ax.add_patch(patches.Wedge((0,0), zone_r, 0, 15, color='#E74C3C', alpha=0.15)) # 右
    ax.text(1.3, -0.35, "End-fire", color='#922B21', fontsize=30, ha='center', fontweight='bold')

    # --- 7. 声源与声波 ---
    ax.plot([0, src_x], [0, src_y], color='#2980B9', linestyle='-.', linewidth=1.5, zorder=2)
    r_label_x = src_x * 0.52 - 0.10
    r_label_y = src_y * 0.52 + 0.20
    ax.text(r_label_x, r_label_y, r"$R=1\,\mathrm{m}/2\,\mathrm{m}$",
        color='#2980B9', fontsize=24, fontweight='bold', ha='right')
    ax.plot([src_x * 0.50, r_label_x - 0.40], [src_y * 0.50, r_label_y - 0.10],
            color='#2980B9', linewidth=1.2, zorder=3)

    # 0° 与声源连线夹角标注 (theta)
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
    
    # 声波 (靠近声源)
    prop_dir = math_angle + 180
    for r in np.linspace(0.3, 0.7, 6):
        arc = patches.Arc((src_x, src_y), 2*r, 2*r, angle=0.0, 
                          theta1=prop_dir - 10, theta2=prop_dir + 10, 
                          color='#C0392B', linewidth=2.5, alpha=0.7)
        ax.add_patch(arc)

    # --- 8. 麦克风阵列 (d 标注) ---
    vis_width = 0.8
    mic_x = np.linspace(-vis_width/2, vis_width/2, 4)
    ax.plot(mic_x, np.zeros_like(mic_x), color='black', linewidth=3, zorder=10)
    ax.scatter(mic_x, np.zeros_like(mic_x), color='black', s=120, zorder=11)
    # d 标注 (只保留一个) + 清晰引出线
    dim_y = -0.25
    gap = -0.02
    overhang = -0.06
    ax.plot([mic_x[1], mic_x[1]], [gap, dim_y + overhang], color='black', lw=0.8)
    ax.plot([mic_x[2], mic_x[2]], [gap, dim_y + overhang], color='black', lw=0.8)
    ax.annotate("", xy=(mic_x[1], dim_y), xytext=(mic_x[2], dim_y),
                arrowprops=dict(arrowstyle='<|-|>', color='black', lw=1.0, mutation_scale=10))
    ax.text((mic_x[1]+mic_x[2])/2, dim_y - 0.20, r"$d=3.5\,\mathrm{cm}$",
            ha='center', va='top', fontsize=30, color='black', fontweight='bold')

    


    # --- 9. 房间尺寸标注 (调用通用函数) ---
    
    # 顶部 5m (向上引出 0.8)
    draw_eng_dim((-2.5, 3.5), (2.5, 3.5), "5 m", offset_dist=0.7)
    
    # 左侧 4m (向左引出 0.8)
    draw_eng_dim((-2, 0), (-2, 4), "4 m", offset_dist=-0.7)

    # --- 11. 装饰 ---
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 视野调整
    plt.xlim(-4.2, 4.2)
    plt.ylim(-1.5, 5.5) # 顶部留给 5m 标注

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    draw_engineering_scenario()