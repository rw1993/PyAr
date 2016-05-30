# PyAR

@(自回归模型)[数据缺失|在线训练]

**PyAR**是一个Python实现的关于面向缺失数据的在线AR模型训练算法包，包含了如下预测器：
 
- **arls_impute_predicter** ：这个预测器并未采用在线算法，而是不断迭代梯度下降、估计方差两个过程，直至收敛，来估计白噪声服从正太分布的AR模型的参数与噪声方差。
- **expand_predicter** ：该预测器基于根据缺失观察结构动态扩展缺失数据的在线梯度下降算法，理论证明详尽，实验效果也很理想，但是由于算法固有的时间复杂度，只能处理参数个数非常有限的AR模型。
- **e_expand_predicter** ：该预测器是基于expand_predicter改进的，可以用于处理参数较多的AR模型，但运行的时间复杂度与时间序列长度相关。
- **similar_expand_predicter**: 该预测器是基于expand_predicter改进的，降低了原来算法的时间复杂度，但是其理论证明尚未明确，实验中也取得了错的效果。
- **ogd_impute_predicter**:该预测器使用简单的在线梯度下降方法，并根据之前的预测进行缺失数据的填补。
- **kalman_impute_predicter**:该预测器使用卡尔曼滤波器的方法，并根据之前的预测对缺失的数据进行填补。
- **yule_walker_impute_predicter**:该预测器使用使用解yule-walker等式的方法对均值为0的AR时间序列进行预测。
- - -------------------
##1、基本的使用

对于上面提到的预测器，我们统一实现了predict_and_fit函数（arls_impute_predicter略有不同）,在实例化predicter之后，我们可以调用其predict_and_fit函数，闯入数值或者代表缺失值的"*"号，便会返回预测值。
##2、参数与属性的解释
对于每种预测器，都会存在一些公共或者特别的参数，在此解释：
- **p**：代表AR模型的自回归参数个数。
- **time_series**:仅在arls_impute_predicter中出现，代表闯入一个一维的列表，列表中是数值或者“*”。
- **missing_indexs**：仅仅在arls_impute_predicter中出现，代表缺失值的位置。
- **stop_error**:仅仅在arls_impute_predicter中出现，代表迭代终止的条件(默认为0.005)。
- **max_iter_time**:仅仅在arls_impute_predicter中出现，代表迭代次数上限。
- **xs**：预测器接受到的观测值的列表。
- **errors**：预测器预测值域观察值之差。
- **missing_ability**：在带expand的预测器初始化时被闯入，与p仪器决定该预测器的d值。
- **learning_rate**: 训练的学习速度。
- **min_ob**：预测器初始化后产生，决定预测器接受多少次观察之后才可以开始预测。
- **no_missing_in_min_ob**：有这个标记的预测器，在前min_ob数量的观测中不能接受缺失数据。
##3、依赖
- **sklearn**
- **pykalman**
- **numpy**
- **scipy**
- **matplotlib**
