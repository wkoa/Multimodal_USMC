from libs.utils import *

classes = ["Wood", "Biodegradable", "Polymers", "Ceramics", "Glasses", "Stones", "Metals", "Composites"]

confuse_matrix = [[0.16634358,0.60484114,0.56602495,1.03682315,0,0.55181846,1.08696955,0.87557604],
 [0.04545785,0.16671086,0.15184619,0.33248882,0,0.15069638,0.29102072,0.22672811],
[0.04732701,0.17477538,0.16079627,0.3274466,0,0.15714799,0.32764082,0.2718894],
[0.01326048,0.04799768,0.04540488,0.08827931,0,0.04309694,0.08718585,0.08663594],
[0.,0.,0.,0.,0,0.,0.,0.],
[0.05112209,0.18467513,0.17378383,0.32282973,0,0.16881533,0.32777657,0.23963134],
[0.02272588,0.0818099,0.0778222,0.14850658,0,0.07452654,0.14547931,0.1281106],
[0.01128028,0.04047889,0.03895327,0.05287506,0,0.0369241,0.0793932 ,0.05437788]]

confuse_matrix = np.array(confuse_matrix)

plot_confuse_matrix(confuse_matrix, classes)