from libs.utils import visualize_log, visualize_log_mtut
import os

# log_file = "log/CBMDataset_sound_accelDFT_macroImage_resnet50_xvector_Early Fusion.log"
# log_file_ = os.path.splitext(log_file)[0]
# for i in range(5):
#     i_log_file = log_file_ + "_{:}".format(i) + ".log"
#     visualize_log(i_log_file, smooth_weight=0.6)
log_fn = os.path.join('log',
                      'UNTIL_CONVERGE_False',
                      'frictionForce_macroImage',
                      'MTUT',
                      "CBMDataset_resnet50_xvector_ReLU_2048_SSALoss_False_2.log")
if "MTUT" in log_fn:
    visualize_log_mtut(log_fn, list_modality=['frictionForce', 'macroImage'], lens=200, aux_loss=True)
else:
    visualize_log(log_fn, lens=200)
