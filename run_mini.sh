epsilons=$1
python main_yyf.py --attack_name LinfMomentumIterativeFastGradientMethod --epsilons $epsilons  
python main_yyf.py --attack_name L2ProjectedGradientDescentAttack --epsilons $epsilons  
python main_yyf.py --attack_name LinfDeepFoolAttack --epsilons $epsilons  
python main_yyf.py --attack_name LinfRepeatedAdditiveUniformNoiseAttack --epsilons $epsilons 
python main_yyf.py --attack_name LinearSearchBlendedUniformNoiseAttack --epsilons $epsilons  
python main_yyf.py --attack_name L2PGD --epsilons $epsilons  