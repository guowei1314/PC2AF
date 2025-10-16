'''
实验环境说明：
- 操作系统：Ubuntu 20.04.6 LTS
- 硬件配置：
  - CPU：Intel Xeon Gold 6248R
  - GPU：2 × NVIDIA GeForce RTX 3090（24GB/张）
  - 内存：256GB
- 软件版本：
  - Python 3.8
  - PyTorch + CUDA

注意：
1. 在不同软硬件环境下运行，实验结果可能存在差异。
2. 文件PC2AF.py已加密为PC2AF.so，当前代码可直接运行（支持使用其他数据集），完整代码将在论文录用后公开。
3. 如有任何问题，请联系：guowei123@email.swu.edu.cn
'''
'''

Experimental Setup:
- OS: Ubuntu 20.04.6 LTS
- Hardware:
  - CPU: Intel Xeon Gold 6248R
  - GPU: 2 × NVIDIA GeForce RTX 3090 (24GB each)
  - RAM: 256GB
- Software:
  - Python 3.8
  - PyTorch with CUDA

Notes:
1. Results may vary under different hardware/software configurations.
2. PC2AF.py has been encrypted as PC2AF.so. The code is ready to run (compatible with other datasets) and will be decrypted upon paper acceptance.
3. For any issues, please contact: guowei123@email.swu.edu.cn
'''

from util_funs import *
from PC2AF import PC2AF_clustering


if __name__ == "__main__":
    data_path = "YaleA_mtv.mat"
    X, gt = load_data_from_mat(data_path) #X^v: dv*N
    k = 4  # m=k*c
    beta = 0.001
    gamma = 0.05
    Z_star, gt_tensor, Clus_num = PC2AF_clustering(X, gt, k, beta, gamma)
    result = perform_clustering(Z_star, gt_tensor, Clus_num)
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} fscore = {:.4f}, precision = {:.4f} recall = {:.4f} ari={:.4f}'.format(result[0], result[1], result[2], result[3], result[4], result[5], result[6]))