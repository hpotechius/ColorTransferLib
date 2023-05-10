from sklearn.cluster import KMeans
# from scipy.stats import multivariate_normal
# from scipy.optimize import minimize
from jax.scipy.optimize import minimize
from scipy.optimize import fmin_cg
from scipy.stats import multivariate_normal
import jax.numpy as jnp

src = cv2.imread("data/images/rose.jpg")
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
ref = cv2.imread("data/images/hyacinth.jpg")
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)
src = cv2.resize(src, (src.shape[1] // 6, src.shape[0] // 6))
ref = cv2.resize(ref, (ref.shape[1] // 6, ref.shape[0] // 6))

# src = np.float32(src) / 255.0
# ref = np.float32(ref) / 255.0
src = np.float64(src)
ref = np.float64(ref)

new_AoW = np.array([
    1.0029e+00, 2.8541e-03, -2.5294e-03, 2.8326e-03, 1.0030e+00,
    -2.2485e-03, 2.8381e-03, 2.9706e-03, 9.9756e-01, 2.7760e-03,
    -2.0686e-03, -2.7232e-03, -2.8570e-03, -2.9683e-03, 2.4483e-03,
    -2.8541e-03, -2.9798e-03, 2.3905e-03, -2.8377e-03, -2.9781e-03,
    2.4364e-03, -2.8106e-03, -2.9614e-03, 2.5341e-03, -2.7823e-03,
    -2.9115e-03, 2.6131e-03, -2.8630e-03, -2.9585e-03, 2.4685e-03,
    -2.8608e-03, -2.9731e-03, 2.4177e-03, -2.8423e-03, -2.9692e-03,
    2.4689e-03, -2.8094e-03, -2.9383e-03, 2.5697e-03, -2.7751e-03,
    -2.8301e-03, 2.6442e-03, -2.8602e-03, -2.8959e-03, 2.5521e-03,
    -2.8569e-03, -2.9167e-03, 2.5395e-03, -2.8371e-03, -2.8867e-03,
    2.5804e-03, -2.8019e-03, -2.7270e-03, 2.6441e-03, -2.7654e-03,
    -1.1359e-04, 2.6903e-03, -2.8490e-03, -4.2192e-04, 2.6375e-03,
    -2.8427e-03, -2.6218e-04, 2.6476e-03, -2.8221e-03, 4.7583e-04,
    2.6766e-03, -2.7887e-03, 1.6798e-03, 2.7088e-03, -2.7555e-03,
    2.2898e-03, 2.7305e-03, -2.8331e-03, 1.7203e-03, 2.6935e-03,
    -2.8238e-03, 2.2584e-03, 2.7087e-03, -2.8038e-03, 2.6033e-03,
    2.7286e-03, -2.7756e-03, 2.7400e-03, 2.7455e-03, -2.7488e-03,
    2.7794e-03, 2.7551e-03, -2.8670e-03, -2.9750e-03, 2.4141e-03,
    -2.8659e-03, -2.9878e-03, 2.3060e-03, -2.8443e-03, -2.9879e-03,
    2.3652e-03, -2.8037e-03, -2.9731e-03, 2.5242e-03, -2.7645e-03,
    -2.9179e-03, 2.6279e-03, -2.8783e-03, -2.9683e-03, 2.4249e-03,
    -2.8813e-03, -2.9852e-03, 2.3069e-03, -2.8557e-03, -2.9829e-03,
    2.4010e-03, -2.8007e-03, -2.9512e-03, 2.5774e-03, -2.7480e-03,
    -2.7895e-03, 2.6712e-03, -2.8761e-03, -2.8893e-03, 2.5492e-03,
    -2.8788e-03, -2.9202e-03, 2.5256e-03, -2.8515e-03, -2.8495e-03,
    2.5996e-03, -2.7886e-03, -2.0479e-04, 2.6879e-03, -2.7297e-03,
    1.5657e-03, 2.7302e-03, -2.8601e-03, 4.9612e-04, 2.6633e-03,
    -2.8561e-03, 1.6189e-03, 2.6864e-03, -2.8257e-03, 2.6414e-03,
    2.7300e-03, -2.7668e-03, 2.8291e-03, 2.7620e-03, -2.7157e-03,
    2.8465e-03, 2.7724e-03, -2.8377e-03, 2.6128e-03, 2.7226e-03,
    -2.8264e-03, 2.8288e-03, 2.7478e-03, -2.7960e-03, 2.9098e-03,
    2.7742e-03, -2.7505e-03, 2.9300e-03, 2.7889e-03, -2.7134e-03,
    2.9237e-03, 2.7904e-03, -2.8688e-03, -2.9651e-03, 2.4751e-03,
    -2.8671e-03, -2.9819e-03, 2.4096e-03, -2.8395e-03, -2.9829e-03,
    2.4617e-03, -2.7829e-03, -2.9652e-03, 2.5857e-03, -2.7334e-03,
    -2.8748e-03, 2.6698e-03, -2.8848e-03, -2.9511e-03, 2.4915e-03,
    -2.8939e-03, -2.9789e-03, 2.3960e-03, -2.8596e-03, -2.9722e-03,
    2.5143e-03, -2.7663e-03, -2.9016e-03, 2.6628e-03, -2.6908e-03,
    -3.0391e-04, 2.7250e-03, -2.8852e-03, -2.6738e-03, 2.6137e-03,
    -2.9001e-03, -2.6483e-03, 2.6187e-03, -2.8642e-03, 2.1638e-03,
    2.7238e-03, -2.7358e-03, 2.8288e-03, 2.7821e-03, -2.6470e-03,
    2.8383e-03, 2.7877e-03, -2.8651e-03, 2.5612e-03, 2.7179e-03,
    -2.8649e-03, 2.9040e-03, 2.7679e-03, -2.8157e-03, 2.9720e-03,
    2.8244e-03, -2.6930e-03, 2.9762e-03, 2.8371e-03, -2.6268e-03,
    2.9607e-03, 2.8222e-03, -2.8365e-03, 2.8814e-03, 2.7612e-03,
    -2.8207e-03, 2.9528e-03, 2.7985e-03, -2.7703e-03, 2.9761e-03,
    2.8302e-03, -2.6911e-03, 2.9789e-03, 2.8385e-03, -2.6481e-03,
    2.9700e-03, 2.8277e-03, -2.8585e-03, -2.9019e-03, 2.5882e-03,
    -2.8526e-03, -2.9322e-03, 2.5841e-03, -2.8194e-03, -2.9233e-03,
    2.6270e-03, -2.7557e-03, -2.8437e-03, 2.6851e-03, -2.7027e-03,
    -4.3955e-04, 2.7226e-03, -2.8731e-03, -2.7360e-03, 2.6303e-03,
    -2.8762e-03, -2.7515e-03, 2.6479e-03, -2.8313e-03, -1.0982e-04,
    2.7185e-03, -2.7037e-03, 1.7431e-03, 2.7678e-03, -2.6178e-03,
    2.4424e-03, 2.7795e-03, -2.8745e-03, 2.1912e-03, 2.7130e-03,
    -2.8878e-03, 2.8898e-03, 2.7792e-03, -2.8309e-03, 2.9821e-03,
    2.8571e-03, -2.5941e-03, 2.9806e-03, 2.8588e-03, -2.5189e-03,
    2.9587e-03, 2.8337e-03, -2.8545e-03, 2.9049e-03, 2.7733e-03,
    -2.8485e-03, 2.9786e-03, 2.8385e-03, -2.7481e-03, 2.9956e-03,
    2.8970e-03, -2.4626e-03, 2.9948e-03, 2.8924e-03, -2.4843e-03,
    2.9840e-03, 2.8580e-03, -2.8251e-03, 2.9442e-03, 2.7939e-03,
    -2.7997e-03, 2.9787e-03, 2.8362e-03, -2.7158e-03, 2.9903e-03,
    2.8689e-03, -2.5826e-03, 2.9910e-03, 2.8729e-03, -2.5627e-03,
    2.9840e-03, 2.8535e-03, -2.8392e-03, -4.4833e-04, 2.6725e-03,
    -2.8269e-03, -3.6495e-04, 2.6898e-03, -2.7918e-03, 1.0795e-04,
    2.7188e-03, -2.7364e-03, 1.1032e-03, 2.7444e-03, -2.6928e-03,
    1.9616e-03, 2.7576e-03, -2.8469e-03, 1.6573e-03, 2.7150e-03,
    -2.8352e-03, 2.5373e-03, 2.7502e-03, -2.7818e-03, 2.8448e-03,
    2.7895e-03, -2.6786e-03, 2.8924e-03, 2.8074e-03, -2.6163e-03,
    2.8795e-03, 2.8044e-03, -2.8449e-03, 2.8494e-03, 2.7662e-03,
    -2.8311e-03, 2.9577e-03, 2.8184e-03, -2.7467e-03, 2.9844e-03,
    2.8622e-03, -2.5613e-03, 2.9849e-03, 2.8662e-03, -2.5113e-03,
    2.9722e-03, 2.8455e-03, -2.8283e-03, 2.9447e-03, 2.7993e-03,
    -2.8015e-03, 2.9822e-03, 2.8492e-03, -2.6819e-03, 2.9937e-03,
    2.8884e-03, -2.4534e-03, 2.9939e-03, 2.8897e-03, -2.4703e-03,
    2.9860e-03, 2.8636e-03, -2.8056e-03, 2.9581e-03, 2.8091e-03,
    -2.7707e-03, 2.9816e-03, 2.8450e-03, -2.6807e-03, 2.9904e-03,
    2.8715e-03, -2.5569e-03, 2.9913e-03, 2.8753e-03, -2.5460e-03,
    2.9855e-03, 2.8584e-03])

# AAA, ooo, WWW = jnp.split(new_AoW, [9, 12])
# W_temp = WWW.reshape(125, 3).T
# print(W_temp.T.reshape(3, 125).flatten())
# exit()

new_AoW2 = np.array([
    8.8514e-01, 4.5808e-01, - 4.0497e-01,
    3.5170e-01, 9.5822e-01, - 2.2160e-01,
    4.9187e-01, - 2.6667e-01, 9.6687e-01,

    -1.7489e-02, -9.9708e-03, 2.0749e-02,

    -8.9005e-02, 3.7674e-02, 1.8872e-01,
    -2.4130e-01, 1.5013e-01, 2.5446e-01,
    6.2058e-02, 1.7819e-02, -1.8946e-01,
    1.7243e-02, 1.4454e-02, -3.9637e-02,
    1.1384e-02, 2.4031e-02, -2.6001e-02,
    -4.2974e-02, 1.5466e-02, 1.0543e-01,
    -8.5008e-02, 5.7508e-02, 1.0891e-01,
    2.9654e-02, 4.5374e-04, -6.5940e-02,
    9.3483e-03, 4.2811e-04, -9.2312e-03,
    1.6878e-03, 5.0055e-03, 6.5157e-04,
    8.1220e-03, 4.3475e-03, -1.3895e-02,
    1.0015e-02, 9.8726e-04, 1.6549e-02,
    2.3837e-03, 2.3905e-03, 2.6965e-03,
    -2.5173e-03, 1.1392e-03, 2.8219e-03,
    3.2935e-03, 5.2481e-03, -4.0503e-03,
    2.6538e-03, 5.1080e-03, -7.6041e-03,
    -1.4708e-03, -1.2954e-04, 2.0517e-02,
    -7.5962e-04, 4.0664e-04, 8.9233e-03,
    -4.0201e-03, -2.6329e-03, 4.7430e-03,
    1.6330e-03, 2.8802e-03, -1.7393e-03,
    1.1953e-02, 1.5655e-02, -3.3362e-02,
    1.9300e-03, 1.9255e-03, -2.8503e-03,
    3.1915e-03, 3.2356e-03, -6.4013e-03,
    1.6882e-03, 2.2939e-03, -4.2357e-03,
    9.8682e-03, 9.3445e-03, -1.2302e-02,
    2.8886e-02, 4.3581e-02, 4.8040e-02,
    -1.9356e-01, 5.9325e-02, 1.2392e-01,
    4.1295e-01, -1.0217e-01, -1.5072e-01,
    1.4767e-02, -1.3473e-01, -2.6073e-02,
    -3.5753e-02, -3.3756e-02, 4.0356e-02,
    3.5017e-01, 3.8796e-02, -2.2570e-01,
    9.4488e-02, -1.1745e-01, -4.7770e-01,
    -1.0313e-01, -6.4841e-02, 3.2061e-01,
    5.4572e-02, -4.8449e-02, 2.6979e-02,
    -2.9379e-02, -3.1859e-02, 4.3729e-02,
    2.4894e-02, -1.2158e-02, -6.3904e-02,
    -6.2971e-02, -9.6858e-02, -3.3684e-01,
    8.1108e-03, -1.6066e-02, -1.3401e-01,
    1.6632e-02, 1.5302e-02, -2.1819e-03,
    7.0041e-03, -2.9020e-04, -2.3148e-03,
    -1.9004e-03, -1.1363e-03, 1.1262e-02,
    -4.2842e-03, -9.3463e-04, 2.4553e-02,
    3.4357e-03, -4.7845e-04, 4.4403e-04,
    -3.4065e-03, -4.2582e-03, 3.0112e-03,
    2.5972e-03, -1.4262e-03, -1.2478e-03,
    4.2301e-03, 1.5015e-03, -8.2192e-03,
    5.5871e-05, -3.0924e-03, 6.3528e-03,
    6.2982e-03, -2.0019e-03, -9.8262e-03,
    2.4027e-03, -2.5781e-03, -5.4114e-03,
    6.7961e-03, 1.4854e-03, -4.1314e-03,
    9.7103e-02, 1.1386e-02, -1.0449e-01,
    -6.6212e-02, -4.4384e-02, 2.7628e-01,
    -2.1338e-02, 7.6595e-02, 4.1152e-01,
    -1.5690e-01, -2.0811e-01, 1.9394e-01,
    -2.1847e-02, 1.4097e-01, -7.9117e-02,
    9.8981e-02, 3.7742e-03, 4.0114e-02,
    -4.9962e-01, -2.6965e-01, -1.3877e-01,
    3.7692e-01, 4.0836e-01, 1.0011e-01,
    -2.4145e-02, 2.5584e-03, 8.5001e-03,
    -3.8029e-02, 1.8363e-02, -9.0357e-02,
    -1.9858e-01, 2.0583e-02, 2.9736e-01,
    3.4319e-01, 2.8943e-02, -2.8367e-01,
    -3.4024e-01, -1.9309e-01, -1.4061e-02,
    -1.1827e-01, 2.1758e-02, 1.4168e-01,
    -8.6234e-03, -2.0395e-02, -2.0818e-02,
    4.4245e-03, 9.6936e-03, 3.4866e-03,
    -9.9273e-03, -2.3120e-02, -4.6302e-02,
    -1.5631e-01, 2.9105e-02, 3.4049e-01,
    -3.3424e-02, 2.4572e-02, 1.3011e-01,
    -8.8487e-03, -1.6777e-02, -4.3825e-03,
    9.3463e-03, 5.4824e-03, -1.2205e-02,
    5.3071e-03, 2.3820e-03, 6.2914e-03,
    2.1887e-02, 3.7483e-03, -2.2014e-02,
    1.3224e-03, -1.1083e-02, -1.9047e-02,
    3.1637e-03, -7.4095e-03, -1.4856e-02,
    6.8787e-03, 2.5179e-03, -4.2178e-02,
    2.4294e-03, -2.8080e-04, -9.7485e-03,
    3.7920e-02, 3.0801e-02, -2.8694e-02,
    3.8114e-02, 6.6373e-02, -8.3987e-02,
    5.4518e-02, 4.4054e-02, -1.0205e-01,
    -1.2739e-01, 2.2640e-02, 2.3138e-01,
    -1.6226e-01, 6.8993e-02, 4.3600e-01,
    4.9857e-02, -6.9827e-03, -1.5421e-01,
    -3.4097e-01, -1.6165e-01, -2.1402e-01,
    -3.4390e-02, -2.2098e-01, -1.5819e-01,
    -1.3555e-01, 4.3563e-02, 2.7084e-01,
    3.1641e-01, -8.5699e-02, -7.7754e-01,
    4.7436e-01, 1.5568e-01, 6.1109e-02,
    2.3145e-01, 1.9988e-01, -7.8599e-02,
    -1.6600e-01, -3.5354e-02, 1.1152e-01,
    4.7575e-03, 1.1613e-02, -1.5873e-03,
    4.4332e-02, -2.3600e-01, -2.8592e-01,
    3.2753e-01, 2.0857e-01, -8.1921e-02,
    6.9401e-01, 7.2999e-02, -5.6137e-01,
    -9.7338e-02, -5.2577e-02, 8.8320e-02,
    9.7857e-03, 8.8207e-03, -8.7140e-03,
    -4.4992e-02, -7.7496e-03, 4.5217e-02,
    -1.7626e-01, -9.0288e-03, 1.2114e-01,
    -1.9483e-01, -1.2290e-01, 1.6843e-01,
    -2.6940e-02, 1.7142e-02, 6.1338e-02,
    1.2833e-02, 1.4462e-02, -4.7809e-02,
    -9.1313e-05, 1.0177e-04, -9.5759e-03,
    -2.0947e-03, 8.7142e-04, -4.9941e-04,
    3.4218e-03, -3.8162e-03, 2.1437e-03,
    2.0996e-02, 1.8090e-03, -1.2859e-02,
    7.3965e-03, 4.9605e-03, -1.3283e-02,
    -1.0461e-02, -5.0235e-03, 8.8482e-03,
    -4.0083e-03, -1.2449e-02, -2.4140e-03,
    -6.8671e-02, -2.1324e-02, 8.3327e-02,
    -8.0802e-02, -1.4771e-02, 9.4362e-02,
    4.4380e-03, 1.1724e-02, -4.1739e-03,
    4.6352e-02, -1.0037e-01, -1.4732e-01,
    2.9465e-01, -1.4539e-02, -3.3444e-01,
    1.4008e-01, 1.2411e-01, -9.5448e-02,
    -6.2528e-02, -1.4272e-02, 5.1545e-02,
    1.1916e-02, 1.3326e-02, -8.3173e-03,
    -6.6387e-02, -1.1771e-01, -3.0507e-02,
    -6.8817e-01, -3.6545e-02, 4.1717e-01,
    -4.4534e-04, 8.3582e-02, 1.2345e-01,
    6.8095e-02, 8.4279e-02, 4.5248e-02,
    2.5109e-02, 1.6219e-02, -2.4885e-02,
    -5.7622e-03, -4.4920e-04, 1.2918e-02,
    -1.4412e-01, -5.2675e-02, 1.1065e-01,
    2.8509e-02, 5.5190e-02, 1.6208e-01,
    1.0003e-01, 9.3987e-02, 1.4709e-02
])

new_AoW3 = np.array([
    #  0.460313   0.729193   4.550112
    # - 2.223041 - 2.341576 - 0.143447
    #   - 0.381504   0.703904 - 0.730884

    0.460313, - 2.223041, - 0.381504,
    0.729193, - 2.341576, 0.703904,
    4.550112, - 0.143447, - 0.730884,

    -0.098699, -0.020874, 0.033157,

    -4.19315, 0.657097, 3.10492,
    -0.163091, -1.66035, -1.48447,
    8.57981, 6.34174, -0.257751,
    1.28687, 1.08942, -0.361447,
    2.03618, 0.179852, -1.57428,
    -0.975077, 0.708736, 0.549263,
    -1.78096, -1.67376, -0.534329,
    2.53563, 2.28146, -0.256172,
    -0.662501, -0.102884, -0.180683,
    0.336273, -0.441318, -0.735163,
    1.19353, 0.946102, 0.129477,
    -2.94556, -1.2295, 1.6436,
    -3.31508, -1.67941, 1.25454,
    -1.46107, -0.8575, 0.493876,
    0.343028, -0.315097, -0.210365,
    2.66842, 1.2637, -0.848272,
    -1.51595, -0.513782, 0.898379,
    -2.82402, -1.07823, 1.36282,
    -1.15647, -0.417185, 1.01155,
    0.72433, 0.110312, 0.0174402,
    4.76214, 2.41232, -2.04081,
    0.818538, 0.892183, -0.403356,
    -0.910408, 0.184579, 0.483666,
    -0.252686, 0.225462, 0.463644,
    1.18879, 0.417966, -0.5799,
    2.58083, -0.166642, -2.76615,
    -1.43431, -3.01891, -1.00108,
    -1.63816, -2.12505, 3.00235,
    0.279093, 2.8947, -2.94325,
    -1.49177, -1.38629, 0.254871,
    8.18675, 2.78321, -3.90421,
    -2.05937, 2.03728, 1.61669,
    -0.754409, -2.98918, -4.13951,
    -0.791843, 0.427136, -2.05793,
    -2.25076, -1.98489, 0.954248,
    1.70995, 1.07238, 0.746514,
    -5.04841, -3.56762, 3.3996,
    -8.21829, -5.34478, 1.55532,
    -0.27128, -0.760662, -0.116676,
    0.18744, -0.261118, 1.06105,
    2.05495, 0.235393, -1.04248,
    -2.88811, -1.2307, 2.12582,
    -4.63698, -1.48066, 2.11649,
    -0.767022, 0.245805, 2.66129,
    0.808481, 0.591993, 1.91697,
    2.60822, 1.28119, -1.55316,
    -2.16282, -0.335517, 0.730472,
    -4.4005, -0.927239, 2.12819,
    -2.33439, -0.472589, 2.72676,
    -0.168051, -0.138318, 1.1385,
    0.681662, 0.51925, 1.14429,
    4.61284, 2.02763, -4.41612,
    -0.642443, -2.03884, 0.731348,
    -1.41212, -0.107656, 2.77515,
    -0.758961, -0.576534, -2.02008,
    -10.5104, 0.290046, 13.1814,
    10.3888, 5.97495, -5.66102,
    -0.806007, -1.11797, -3.0954,
    1.14221, -1.1264, -1.87704,
    -2.84919, -2.96571, 0.259265,
    9.59926, -1.58492, -11.5074,
    9.54753, 7.7595, -0.870047,
    0.271673, -2.20221, -5.46644,
    12.9186, 3.5068, -5.85418,
    2.56549, 1.48624, 1.24366,
    2.07451, -1.16188, -1.89805,
    0.832965, 0.234886, 6.6419,
    -4.2366, 0.576848, -6.32232,
    2.81056, 3.72833, 9.43125,
    3.90272, 2.89085, 2.6412,
    2.03236, 0.692915, -2.0294,
    -4.48118, -1.81526, 0.84714,
    -9.54326, -3.14186, 3.1757,
    -3.35508, -2.31631, 5.00853,
    -0.126459, -0.948246, 0.704561,
    -0.623839, -1.21309, 0.205514,
    0.966381, 0.539281, -0.100192,
    1.48494, 1.83295, 0.871819,
    -0.346308, -0.0400972, -0.644843,
    1.32528, 0.752332, -3.29744,
    -5.09324, -5.77678, -2.60813,
    -12.049, -1.97696, 13.3615,
    7.46769, 8.96419, 5.22016,
    -2.98526, -3.00914, -2.51701,
    -1.78894, -1.90244, 1.50113,
    -13.8215, -9.90345, 1.29032,
    5.3357, 2.36628, -0.865826,
    2.08267, 0.389173, -2.21669,
    0.265582, 0.773393, -4.04586,
    4.34574, 4.35766, -3.13502,
    3.7306, -0.0177066, -5.74994,
    8.36544, 1.23965, -6.8622,
    9.99709, 6.38522, -2.21104,
    1.82782, 0.797353, -3.0775,
    5.92973, 7.19322, -6.58427,
    2.9898, 1.57894, -2.89505,
    -4.31219, -2.61329, 0.643958,
    -12.0258, -9.0373, 4.40557,
    2.28619, -2.2333, -3.1542,
    -3.24136, -3.71939, -4.68735,
    -0.380637, -1.21325, 1.0525,
    -1.12355, -0.130378, 2.76482,
    -0.419611, 0.530679, 2.74075,
    -0.252521, -0.334518, 1.07567,
    1.37667, -0.00154425, -0.521138,
    -4.72054, -2.6168, 0.817298,
    -2.93918, 1.06692, 4.22579,
    -1.41305, 1.41834, 3.98113,
    -2.74876, -1.92292, 1.04505,
    1.40765, 0.379212, -0.639387,
    -0.635664, 0.0593338, -3.79,
    9.30998, 8.59003, -6.52033,
    -16.1116, -9.34289, 13.544,
    -10.0107, -13.8357, 5.08865,
    1.13885, 2.02705, -3.57167,
    4.09404, 2.77888, -4.0254,
    -5.7508, -4.41709, 9.42918,
    10.7726, 6.52896, -12.5162,
    -0.934481, 0.540974, -1.76938,
    1.98436, 6.49657, -2.77136,
    4.3307, 3.66782, -1.41471,
    0.770786, 2.00889, 1.93947,
    -3.16242, -0.916419, 6.42456,
    12.4005, 8.44897, 2.54327,
    -5.17212, -7.24313, 0.695238
])

h = src.shape[0]
w = src.shape[1]

X, Y, Z = np.mgrid[0:256:63.75, 0:256:63.75, 0:256:63.75]
C = torch.vstack((torch.tensor(Z.flatten()), torch.tensor(Y.flatten()), torch.tensor(X.flatten()))).T


def tps(x):
    # vecX = np.full((125, 3), x)
    print(x[0])
    print(x[1])
    print(x[2])
    # exit()
    vecX1 = torch.full((125, 1), x[0])
    vecX2 = torch.full((125, 1), x[1])
    vecX3 = torch.full((125, 1), x[2])
    vecX = torch.hstack((vecX1, vecX2, vecX3))
    print(vecX)
    # exit()
    # print(vecX)
    # exit()
    # vecX = torch.Tensor(125, 3).fill_(x)
    # print(X)
    # exit()
    # X, Y, Z = torch.meshgrid[0:256:63.75, 0:256:63.75, 0:256:63.75]
    # X, Y, Z = jnp.mgrid[0:1.1:0.25, 0:1.1:0.25, 0:1.1:0.25]
    temp = vecX - C
    # print(temp)
    # exit()
    return -torch.linalg.norm(temp, axis=1)


def transfer(A, o, W, x):
    # A = np.array([
    #    [8.8514e-01,   4.5808e-01,   - 4.0497e-01],
    #    [3.5170e-01, 9.5822e-01, - 2.2160e-01],
    #    [4.9187e-01,   - 2.6667e-01,   9.6687e-01]
    # ])

    Y = tps(x)
    Y = torch.tensor(Y)

    # print(A)
    # print(A.shape)
    # print(o)
    # print(o.shape)
    # print(W.dtype)
    # print(W.shape)
    # print(Y.dtype)
    # print(Y.shape)
    # print(x)
    # print(x.shape)
    # exit()

    # print(Y[6])
    Phi = A @ x + o + W @ Y
    # print(A)
    # print(x)
    # print(Phi)
    # exit()

    return Phi  # np.clip(Phi, 0, 255)


"""
def transfer_NP(A, o, W, x):

    Y = tps(x)
    Phi = A.dot(x) + o + W.dot(Y)

    return np.clip(Phi, 0, 255)

# loop over the image, pixel by pixel
AAA, ooo, WWW = np.split(new_AoW3, [9, 12])
#print(AAA.reshape(3, 3))
#exit()

#print(WWW.reshape(125,3).T)
#exit()
for y in range(0, h):
    print(y)
    for x in range(0, w):
        # threshold the pixel
        src[y, x] = transfer_NP(AAA.reshape(3, 3), ooo, WWW.reshape(125, 3).T, src[y, x])
        #exit()
        #print(src[y, x])

src = np.uint8(src)
src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
ref = np.uint8(ref)
ref = cv2.cvtColor(ref, cv2.COLOR_RGB2BGR)

#cv2.imshow("winSrc", src)
cv2.imshow("winRef", ref)
cv2.imshow("winNew", src)
cv2.waitKey(0)
"""

# print(src.dtype)
# exit()

K = 50

src_km = KMeans(n_clusters=K, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
src_h, src_w, src_d = src.shape
src_reshaped = src.reshape(src_h * src_w, src_d)
src_y_km = src_km.fit_predict(src_reshaped)

ref_km = KMeans(n_clusters=K, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
ref_h, ref_w, ref_d = ref.shape
ref_reshaped = ref.reshape(ref_h * ref_w, ref_d)
ref_y_km = ref_km.fit_predict(ref_reshaped)

d = 3
# A_in = np.random.rand(3,3)
# o_in = np.random.rand(3,)
# W_in = np.random.rand(3,125)

A_in = np.identity(3)
o_in = np.zeros((3,))
W_in = np.zeros((3, 125))

ann_steps = 5
# max_iter = 10000
m = 125
# C = np.zeros((125, 3))
h_band = (pow(ann_steps - 1, 2)) * pow(np.linalg.det(src_km.cluster_centers_.T.dot(src_km.cluster_centers_ / m)),
                                       1 / (pow(d, 2)))
# h_band = 0.1
h_band = torch.tensor(h).double()


def multivariate_normalo(x, d, mean, covariance):
    # pdf of the multivariate normal distribution.
    # print(x.dtype)
    # print(mean.dtype)
    # print(covariance)
    x_m = x - mean
    ret = (1. / (torch.sqrt((2 * 3.1415927410125732) ** d * torch.linalg.det(covariance))) * torch.exp(
        -(torch.linalg.solve(covariance, x_m).T @ (x_m)) / 2))
    return ret


def QRE(A, o, W, K, src_u):
    qre = 0
    # print(torch.eye(3))
    # print(torch.tensor(np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])))
    # exit()
    cov = 2 * pow(h, 2) * torch.eye(3, dtype=torch.double)
    # print(cov)
    # exit()
    # cov = torch.tensor(cov)
    # print(cov.dtype)
    for i in range(K):
        for k in range(K):
            mean = transfer(A, o, W, src_u[i]) - transfer(A, o, W, src_u[k])
            # qre += multivariate_normal.logpdf(torch.tensor(np.array([0, 0, 0])), mean=mean, cov=cov) * (1. / K) * (1. / K)
            # qre += multivariate_normalo(torch.zeros(3), 3, mean, cov) * (1.0 / K) * (1.0 / K)
            # print(torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).log_prob(torch.zeros(3)))
            qre += torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).log_prob(torch.zeros(3)) * (
                        1. / K) * (1. / K)
    return qre


def RCE(A, o, W, K, src_u, ref_u):
    rce = 0
    cov = 2 * pow(h, 2) * torch.eye(3, dtype=torch.double)
    # cov = torch.tensor(cov)
    for i in range(K):
        for k in range(K):
            mean = transfer(A, o, W, src_u[k]) - ref_u[i]
            # print(transfer(A, o, W, src_u[k]))
            # rce += multivariate_normal.logpdf(torch.tensor(np.array([0, 0, 0])), mean=mean, cov=cov) * (1. / K) * (1. / K)
            # rce += multivariate_normalo(torch.zeros(3), 3, mean, cov) * (1.0 / K) * (1.0 / K)
            rce += torch.distributions.multivariate_normal.MultivariateNormal(mean, cov).log_prob(torch.zeros(3)) * (
                        1. / K) * (1. / K)
    # print(rce.dtype)
    return rce


def L2E(A, o, W):
    src_u_in = src_km.cluster_centers_
    ref_u_in = ref_km.cluster_centers_
    src_u_in = torch.tensor(src_u_in)
    ref_u_in = torch.tensor(ref_u_in)

    # print(QRE(A, o, W, K, src_u))
    # print(RCE(A, o, W, K, src_u, ref_u))
    # exit()
    # l2e = QRE(A, o, W, K, src_u_in)
    l2e = QRE(A, o, W, K, src_u_in) - 2 * RCE(A, o, W, K, src_u_in, ref_u_in)
    # l2e = QRE(A, o, W, K, src_u_in) - RCE(A, o, W, K, src_u_in, ref_u_in)
    return l2e


def minL2E(AoW):
    # AoW = tf.compat.v1.Session().run(AoW)

    AAA, ooo, WWW = torch.tensor_split(AoW, (9, 12))

    ll = L2E(AAA.reshape(3, 3).T, ooo, WWW.reshape(125, 3).T)
    return ll


def regular(A, o, W, pixel):
    suma = 0
    for c1 in range(0, 3):
        for c2 in range(0, 3):

            def temp_trans(aa, bb, cc):
                # print(torch.stack([aa, bb, cc]))
                yy = torch.zeros((3,))
                print(aa)
                print(bb)
                print(cc)
                yy[0] += aa
                yy[1] += bb
                yy[2] += cc
                print(yy)
                print(yy[0])
                print(yy[1])
                print(yy[2])
                exit()
                # print(torch.tensor([aa, bb, cc]))
                # exit()

                return transfer(torch.tensor(A_in), torch.tensor(o_in), torch.tensor(W_in),
                                [aa, bb, cc])

            # threshold the pixel
            pix1a = torch.tensor(pixel[c1], requires_grad=True)
            if c1 == 0:
                print("Yeps")
                # vect = torch.tensor([pix1a, torch.tensor(pixel[1]), torch.tensor(pixel[2])])
                first_derivative = temp_trans(pix1a, torch.tensor(pixel[1]), torch.tensor(pixel[2]))
            """elif c1 == 1:
                #vect = torch.tensor([torch.tensor(pixel[0]), pix1a, torch.tensor(pixel[2])])
                first_derivative = temp_trans(torch.tensor(pixel[0]), pix1a, torch.tensor(pixel[2]))
            else:
                #vect = torch.tensor([torch.tensor(pixel[0]), torch.tensor(pixel[1]), pix1a])
                first_derivative = temp_trans(torch.tensor(pixel[0]), torch.tensor(pixel[1]), pix1a)"""
            # first_derivative = torch.autograd.grad(torch.tensor([transfer], requires_grad=True), [A,o,W,pix1], create_graph=True)[0]
            # We now have dloss/dx
            # second_derivative = torch.autograd.grad(first_derivative, pix2)[0]

            # transfer(A, o, W, vect)
            first_derivative.backward()

            # pix2 = torch.tensor(pixel[c2], requires_grad=True)
            # second_derivative = first_derivative(pix2)
            # second_derivative.backward()

            # suma = pow(second_derivative, 2)
            # src[y, x] = transfer(A, o, W, x)
            # exit()
            # print(src[y, x])
    return suma


print("WE")
print(regular(torch.tensor(A_in), torch.tensor(o_in), torch.tensor(W_in), src[0, 0]))
exit()


def cback(aoW):
    print("yes")
    return True


# theta = np.concatenate((np.concatenate((A_in.flatten(), o_in)), W_in.flatten()))
# theta = torch.tensor(theta, requires_grad=True)
# print(minL2E(torch.tensor(new_AoW3)))
# exit()

#############################################################
#############################################################
print("START")
lr = 1e-3
n_epochs = 1

theta = np.concatenate((np.concatenate((A_in.flatten(), o_in)), W_in.flatten()))
theta = torch.tensor(theta, requires_grad=True)
# theta = torch.tensor(new_AoW2, requires_grad=True)

optimizer = torch.optim.Adam([theta], lr=lr)

for t in range(5):
    for epoch in range(n_epochs):
        loss = minL2E(theta)
        loss.backward()

        print(loss)
        optimizer.step()
        optimizer.zero_grad()
        # print(theta.grad)
        # print(theta.grad)
        # with torch.no_grad():
        #    theta -= lr * theta.grad

        # theta.grad.zero_()
    h_band = 0.5 * h_band
print(theta)

print("END")
exit()

#############################################################
#############################################################

# print(cost(A_in, o_in, W_in))

"""
print("Wait")
theta = np.concatenate((np.concatenate((A_in.flatten(), o_in)), W_in.flatten()))
@tf.function
def quadratic_with_bfgs():
    return tfp.optimizer.bfgs_minimize(
        minL2E,
        initial_position=theta,
        tolerance=0.1)


results = quadratic_with_bfgs()
exit()
"""

# print(minL2E(theta))
# print(theta)
print(jnp.ones(3).device_buffer.device())

theta = jnp.concatenate((jnp.concatenate((A_in.flatten(), o_in)), W_in.flatten()))
for t in range(5):
    result = minimize(minL2E, theta, method="BFGS", tol=1e-04)
    # result = fmin_cg(minL2E, theta)
    # print(np.split(theta, [9, 12])[1])
    h_band = 0.5 * h_band
    # print(result.fun)
    # print(result.status)
    # print(result.x)
    print(result)

exit()