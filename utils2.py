from tensorflow.python.keras import *
import tensorflow as tf
import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------
#  Global Parameters
# ---------------------

Nt = 64
P = 1   # the normalized transmit power

# ---------------------
#  Functions
# ---------------------

# transfer the phase to complex-valued analog beamformer
def trans_Vrf(temp, num_antennas):
    v_real = tf.cos(temp)
    v_imag = tf.sin(temp)
    vrf = tf.cast(tf.complex(v_real, v_imag), tf.complex64)
    return vrf

def trans_Vrf1x64(temp):
    v_real = tf.cos(temp)
    v_imag = tf.sin(temp)
    vrf = tf.cast(tf.complex(v_real, v_imag), tf.complex64)
    return vrf

def Rate_func(temp):
    h, v, SNR_input = temp
    hv = tf.keras.backend.batch_dot(
        tf.cast(h, tf.complex64), v)
    rate = tf.math.log(tf.cast(1 + SNR_input / Nt * tf.pow(tf.abs(hv), 2), tf.float32)) / tf.math.log(2.0)
    return -rate

# For the simplification of implementation based on Keras, we use a lambda layer to compute the rate
# Thus, the output of the model is actually the loss.
# For the simplification of implementation based on Keras, we use a lambda layer to compute the rate
# Thus, the output of the model is actually the loss.
def Rate_func_MIMO(temp_inputs):
    H_perfect, f_RF, w_RF, SNR_input, Nt_const, Nr_const = temp_inputs
    # f_RF_col shape: (batch_size, Nt, 1)
    f_RF_col = tf.expand_dims(f_RF, axis=-1)
    # w_RF_col shape: (batch_size, Nr, 1)
    w_RF_col = tf.expand_dims(w_RF, axis=-1)

    # Tính w_RF^H (chuyển vị liên hợp của w_RF)
    w_RF_H = tf.linalg.adjoint(w_RF_col)

    # Tính H * f_RF
    H_f_RF = tf.matmul(H_perfect, f_RF_col)

    # Tính w_RF_H * H_f_RF (tử số phức)
    effective_channel_gain_complex = tf.matmul(w_RF_H, H_f_RF)

    # Lấy norm bình phương của tử số: ||w_RF^H H f_RF||^2
    power_signal = tf.square(tf.abs(tf.squeeze(effective_channel_gain_complex, axis=[1, 2])))

    # Mẫu số: Nt * Nr (do ||w_RF^H||^2 = Nr)
    denominator_factor = Nt_const * Nr_const

    # SINR = SNR * power_signal / denominator_factor
    snr_squeezed = tf.squeeze(SNR_input, axis=-1) # shape (batch_size,)
    sinr_val = snr_squeezed * power_signal / denominator_factor

    # Rate = log2(1 + SINR)
    rate = tf.math.log(1.0 + tf.cast(sinr_val, tf.float32)) / tf.math.log(2.0)
    return -rate

def mat_load_train(folder_path):
    print(f"loading data...")
    try:
        # load pcsi
        data_pcsi = sio.loadmat(os.path.join(folder_path, 'pcsi.mat'))
        h = data_pcsi['pcsi']

        # load ecsi
        data_ecsi = sio.loadmat(os.path.join(folder_path, 'ecsi.mat'))
        h_est = data_ecsi['ecsi']
        print('Tải dữ liệu hoàn tất.')
        print(f'Shape của kênh ước tính là: {h_est.shape}')
        return h, h_est
    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy một hoặc nhiều file trong thư mục '{folder_path}'.")
        print(f"Chi tiết lỗi: {e}")
        return None, None
        

def mat_load_test(folder_path):
    print(f"loading data...")
    try:
        # load pcsi
        data_pcsi = sio.loadmat(os.path.join(folder_path, 'pcsi.mat'))
        h = data_pcsi['pcsi']

        # load ecsi
        data_ecsi = sio.loadmat(os.path.join(folder_path, 'ecsi.mat'))
        h_est = data_ecsi['ecsi']
        
        # load AoA
        data_aoa = sio.loadmat(os.path.join(folder_path, 'AoA.mat'))
        AoA = data_aoa['AoA']
        
        # load AoD
        data_aod = sio.loadmat(os.path.join(folder_path, 'AoD.mat'))
        AoD = data_aod['AoD']
        
        # load alpha
        data_alpha = sio.loadmat(os.path.join(folder_path, 'alpha.mat'))
        alpha = data_alpha['alpha']

        print('Tải dữ liệu hoàn tất.')
        print(f'Shape của kênh ước tính là: {h_est.shape}')
        return h, h_est, AoA, AoD, alpha
    except FileNotFoundError as e:
        print(f"LỖI: Không tìm thấy một hoặc nhiều file trong thư mục '{folder_path}'.")
        print(f"Chi tiết lỗi: {e}")
        return None, None, None, None, None

def plot_beam_pattern(beam_vector, num_antennas, title_str, true_angle_deg=None):
    """Vẽ mẫu bức xạ 2D của một vector búp sóng đầy đủ 360 độ."""
    # Thay đổi phạm vi góc từ -180 đến 180 độ để bao phủ 360 độ
    # Hoặc từ 0 đến 360 độ nếu bạn thích (np.linspace(0, 360, 361))
    angles_deg = np.linspace(-180, 180, 361) # 361 điểm cho 1 độ/điểm
    angles_rad = np.deg2rad(angles_deg)
    array_factor = []
    antenna_indices = np.arange(0, num_antennas)

    # Tính toán Array Factor
    for theta in angles_rad:
        # Công thức steering vector cho ULA với d=lambda/2
        steering_vector = np.exp(1j * np.pi * np.sin(theta) * antenna_indices)
        af_value = np.abs(np.vdot(beam_vector, steering_vector))
        array_factor.append(af_value)
        
    array_factor = np.array(array_factor)
    # Chuẩn hóa và chuyển sang dB, đảm bảo không chia cho 0 nếu array_factor rỗng hoặc tất cả 0
    if np.max(array_factor) == 0:
        array_factor_db = np.full_like(array_factor, -np.inf) # Hoặc giá trị dB rất nhỏ
    else:
        array_factor_db = 20 * np.log10(array_factor / np.max(array_factor))

    # Cấu hình và vẽ biểu đồ
    plt.figure(figsize=(10, 10)) # Tăng kích thước để biểu đồ 360 độ rõ ràng hơn
    ax = plt.subplot(111, polar=True)
    ax.plot(angles_rad, array_factor_db, label='Predicted Beam Pattern')

    ax.set_ylim([-40, 5]) # Giới hạn trục bán kính (dB)
    ax.set_theta_zero_location("N") # Hướng 0 độ ở phía trên (North)
    
    # Đặt các đường lưới theo góc để bao phủ 360 độ
    # Từ 0 đến 360 (hoặc -180 đến 180) với bước 30 độ
    # set_thetagrids nhận list các góc độ
    ax.set_thetagrids(np.arange(0, 361, 30)) # Lưới từ 0 đến 360 độ, bước 30
    
    ax.set_yticklabels([]) # Ẩn các nhãn của trục bán kính
    ax.set_title(title_str, fontsize=14, pad=20) # Thêm pad để tiêu đề không bị chồng lên biểu đồ
    
    # Vẽ góc thực
    if true_angle_deg is not None:
        true_angle_rad = np.deg2rad(true_angle_deg)
        # Đảm bảo đường vẽ xuất phát từ tâm (hoặc gần tâm) đến biên
        ax.plot([true_angle_rad, true_angle_rad], [ax.get_ylim()[0], ax.get_ylim()[1]], 
                'r--', linewidth=2, label=f'True Angle ({true_angle_deg:.1f}°)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1)) # Đặt legend ra ngoài nếu cần

    plt.grid(True)
    plt.show()