import sys
import csv
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def de_bruijn(k, n):
    """
    de Bruijn sequence for alphabet k
    and subsequences of length n.
    """
    try:
        # let's see if k can be cast to an integer;
        # if so, make our alphabet a list
        _ = int(k)
        alphabet = list(map(str, range(k)))

    except (ValueError, TypeError):
        alphabet = k
        k = len(k)

    a = [0] * k * n
    sequence = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                sequence.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
    db(1, 1)

    return "".join(alphabet[i] for i in sequence), sequence

def get_2d_PRBA(k1, k2, n1, n2):
    PRBA = np.zeros((n1, n2))
    n = k1 * k2
    _, seq = de_bruijn(2, n)
    shifted_sequence = np.roll(seq, len(seq) // 4)
    row = 0
    col = 0
    for idx in range(1, len(seq)):
        PRBA[row, col] = shifted_sequence[idx] * 255
        row = (row + 1) % n1
        col = (col + 1) % n2
    return PRBA.astype(np.uint8)

def get_img_from_PRBA(PRBA_mat):
    cell_size = 5
    center_pos = cell_size // 2
    center_size = 1
    center_size_half = center_size // 2
    height = PRBA_mat.shape[0] * cell_size
    width = PRBA_mat.shape[1] * cell_size
    PRBA_img = np.zeros((height, width)).astype(np.uint8)

    for r in range(PRBA_mat.shape[0]):
        for c in range(PRBA_mat.shape[1]):
            r_pos = r * cell_size + center_pos
            c_pos = c * cell_size + center_pos
            PRBA_img[r_pos-center_size_half:r_pos+center_size_half+1, c_pos-center_size_half:c_pos+center_size_half+1] = PRBA_mat[r, c]
    
    return PRBA_img

def convert_img2positions(img):
    csv_array = []
    x_array = []
    y_array = []
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            if img[r, c] == 255:
                csv_array.append([c*10 + 5, r*10 + 5])
                x_array.append(c*10 + 5)
                y_array.append(r*10 + 5)
    
    # Plotting dots
    plt.scatter(x_array, y_array, color='blue', label='Dots')

    # Adding labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Scatter Plot of Dots')

    # Displaying the plot
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return csv_array

if __name__ == "__main__":

    k1 = 6
    k2 = 2
    n1 = 63
    n2 = 65

    PRBA_mat = get_2d_PRBA(k1, k2, n1, n2)
    PRBA_mat_PIL = Image.fromarray(PRBA_mat)
    PRBA_mat_PIL.save("./Pattern_mat.png")
    
    PRBA_img = get_img_from_PRBA(PRBA_mat)
    PRBA_img_PIL = Image.fromarray(PRBA_img)
    PRBA_img_PIL.save("./Pattern_img.png")

    cvs_array = convert_img2positions(PRBA_img)
    file_path = 'Pattern_img.csv'
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(cvs_array)
