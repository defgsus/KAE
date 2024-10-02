class DENSE_LAYER_CONST:
    LINEAR_LAYER = "LINEAR"
    KAN_LAYER = "KAN"
    WAVELET_KAN_LAYER = "WAVELET_KAN"
    FOURIER_KAN_LAYER = "FOURIER_KAN"
    KAE_LAYER = "KAE"
    MULTI_KAN_LAYER = "MULTI_KAN"

    LAYER_TYPES = [
        LINEAR_LAYER,
        KAN_LAYER,
        WAVELET_KAN_LAYER,
        FOURIER_KAN_LAYER,
        KAE_LAYER,
        MULTI_KAN_LAYER,
    ]

    TAYLOR_KAN_LAYER_DEFAULT_ORDER = 3