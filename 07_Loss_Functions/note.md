# 常用的损失函数与自定义损失函数
[toc]
> 1. 常用的损失函数
> 2. 自定义损失函数
> 3. 案例讲解

## 01. 常用的损失函数
损失函数就是评估模型中预测值和真实值之间差异的程度，神经网络中优化的目标函数。神经网络优化的过程就是最小化损失函数的过程。

TF中的损失函数在`tf.keras.losses` 中，
官方API地址： https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/losses

常见的损失函数：
* mean_squared_error（平方差误差损失，用于回归，简写为 mse，类实现形式为`MeanSquaredError` 和 `MSE`）
* binary_crossentropy（二元交叉嫡，用于二分类，类实现形式为 `BinaryCrossentropy`）
* categorical_crossentropy（类别交叉嫡，用于多分类，**要求label为onehot编码**，类实现形式为`CategoricalCrossentropy`）
* sparse_categorical_crossentropy（稀疏类别交叉嫡，用于多分类，**要求label为序号编码形式**，类实现形式为`SparseCategoricalCrossentropy`）

## 02. 自定义损失函数

> 两种方式自定义损失函数：
> 1. 函数的实现形式
> 2. 类的实现形式

```python
class MeanSquaredError(tf.keras.losses.Loss):
    """
    MSE的类实现形式：需要继承tf.keras.losses.Loss
    """
    def call (self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))

def MeanSquaredError(y_true, y_pred):
    """
    MSE的函数实现形式
    """
    return tf.reduce_mean(tf.square(y_pred - y_true))

```

### Focal loss损失函数实现

论文地址： Focal Loss for Dense Object Detection

相关讨论： https://www.zhihu.com/question/63581984，主要是针对类别不平衡的问题提出的新型的损失函数。

原始论文的 Focal loss 损失函数针对于二分类，这里改为多分类损失函数 ：
![](media/Focal_Loss.png)

```python
# 多分类的focal loss损失函数
class SparseFocalloss(tf.keras.losses.Loss):
    """
    类实现形式
    """
    def __init__(self, gamma=2.0, alpha=.25, class_num=10):
        self.gamma = gamma
        self.alpha = alpha
        self.class_num = class_num
        super(SparseFocalloss, self).__init_()
    
    def call(self, y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        y_true = tf.one_hot(y_true, depth=self.class_num)
        y_true = tf.cast(y_true, tf.float32)
        loss = -y_true * tf.math.pow(1 - y_pred，self.gamma) * tf.math.log(y_pred)
        loss = tf.math.reduce_sum(loss, axis=1)
        return loss
```
```python
def focal_loss(gamma=2.0, alpha=0.25):
    """
    函数实现形式
    """
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.nn.softmax(y_pred, axis=-1)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
        y_true = tf.cast(y_true, tf.float32)
        loss = -y_true * tf.math.pow(1 - y_pred，gamma) * tf.math.log(y_pred)
        loss = tf.math.reduce_sum(loss, axis=1)
        return loss
    return focal_loss_fixed
```
