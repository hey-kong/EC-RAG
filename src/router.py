from matrix_factorization.model import MFModel

router = MFModel()
router.load("/data/models/mf_model.pth")