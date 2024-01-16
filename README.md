# Intelligent-Security-Competition

### :fish:背景

![image-20240116125537685](E:\【研究生】课程相关\【研一上】\智能安全\Intelligent-Security-Competition\image_pub\image-20240116125537685.png)

通过引入对抗样本，攻击者很容易就可以通过肉眼几乎观察不到的微小扰动，使模型分类失误。

### :rocket:主要贡献

![image-20240116125953695](E:\【研究生】课程相关\【研一上】\智能安全\Intelligent-Security-Competition\image_pub\image-20240116125953695.png)

- 防御模型的集成可以在一定程度上提高整体的鲁棒性，我们决定同时集成来自ARES库的6个防御模型和RoubustBench排行榜上的4个防御模型
- 攻击方法选择了包含L2PGD攻击方法在内的组合攻击方法AutoAttack，用于攻击模型以生成对抗样本
- 二分空间中，最优参数搜索和参数动态适应优化

###  :bike:实验结果

在《 [**Cifar-10 数字世界 无限制对抗攻击竞赛**](http://221.122.70.202:8080/#/competitionDetail?id=11)》中排名TOP1

![image-20240116130509391](E:\【研究生】课程相关\【研一上】\智能安全\Intelligent-Security-Competition\image_pub\image-20240116130509391.png)

### :heavy_exclamation_mark:声明

相关模型权重和关键代码未上传，后续课程禁止复用抄袭本仓库代码
