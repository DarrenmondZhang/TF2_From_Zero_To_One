# 数据集下载链接：
链接：https://pan.baidu.com/s/1nthtU5BPZM9rOVPRrcrznA 
提取码：pq8b
 
# `Dataset`类详解
> 1. Dataset类相关操作
> 2. 如何提升Dataset读取性能( `prefetch`、`interleave`、`map`、`cache`)
> 3. 案例讲解


## 01. Dataset类相关操作

`tf.data.Dataset`类创建数据集，对**数据集实例化**。最常用的如:
* `tf.data.Dataset.from_tensors()`: 创建Dataset对象，合并输入并返回具有单个元素的数据集。
* `tf.data.Dataset.from_tensor_slices()`: 创建一个Dataset对象，输入可以是一个或者多个tensor，若是多个tensor，需要以元组或者字典等形式组装起来。
* `tf.data.Dataset.from_generator()`: 迭代生成所需的数据集，一般数据量较大时使用。
  
> 注:Dataset可以看作是相同类型“元素”的有序列表。在实际使用时，单个“元素”可以是向量，也可以是字符串、图片，甚至是tuple或者dict。

`Dataset`包含了非常丰富的数据转换功能:
* `map(f)`: 对数据集中的每个元素应用函数f，得到一个新的数据集（这部分往往结合`tf.io`进行读写和解码文件，`tf.image`进行图像处理)
* `shuffle(buffer_size)`︰将数据集打乱（设定一个固定大小的缓冲区(Buffer），取出前`buffer_size`个元素放入，并从缓冲区中随机采样，采样后的数据用后续数据替换)
* `repeat(count)`: 数据集重复次数。
* `batch(batch_size)`: 将数据集分成批次，即对每batch_size个元素，使用tf.stack(在第0维合并，成为一个元素;

![](media/dataset类操作.jpg)

* `flat_map()`: 将`map`函数映射到数据集的每一个元素，并将嵌套的`Dataset`压平。
```python
flat_map(map_func)
```
![](media/flat_map使用.jpg)![](media/flat_map使用2.jpg)

* `interleave()`: 效果类似`flat_map`，但可以将不同来源的数据夹在一起。
```python
interleave(
    map_func, cycle_length=None, block_length=None, num_parallel_calls=None,
    deterministic=None
)
```
`interleave()`是`Dataset`的类方法，所以`interleave`是作用在一个`Dataset`上的。

首先该方法会从该Dataset中取出cycle_length个element，然后对这些element 应用 map_func，得到cycle_length个新的Dataset对象。然后从这些新生成的Dataset对象中取数据，每个Dataset对象一次取block_length个数据。当新生成的某个Dataset的对象取尽时，从原Dataset中再取一个element，然后应用 map_func，以此类推。

![](media/interleave使用案例.jpg)

> https://tensorflow.google.cn/api_docs/python/tf/data/Dataset?hl=en#interleave

* `take()`: 截取数据集中的前若干个元素
* `filter`: 过滤掉某些元素。
* `zip`: 将两个长度相同的Dataset横向饺合。
![](media/zip.jpg)
* `concatenate`: 将两个Dataset纵向连接。
![](media/concat.jpg)
* `reduce`: 执行归并操作。

## 02. 提升`Dataset`读取性能

训练深度学习模型常常会非常耗时。模型训练的耗时主要来自于两个部分，一部分来自数据准备，另一部分来自参数迭代。

参数迭代过程的耗时通常依赖于GPU来提升，而数据准备过程的耗时则可以通过构建高效的数据管道进行提升。

以下是一些构建高效数据管道的建议。
1. 使用`prefetch`方法让数据准备和参数迭代两个过程相互并行。
2. 使用`interleave`方法可以让数据读取过程多进程执行，并将不同来源数据夹在一起。
3. 使用`map`时设置num_parallel_calls让数据转换过程多进程执行。
4. 使用`cache`方法让数据在第一个epoch后缓存到内存中，仅限于**数据集不大**情形。

原始方法执行，可以看到执行训练步骤涉及:
1. 打开文件（如果尚未打开)
2. 从文件中获取数据条目
3. 使用数据进行训练。
   
![](media/原始执行训练步骤.jpg)

### 2.1 prefetech

`prefetch`与训练步骤的预处理和模型执行重叠。当模型执行训练步骤时s，输入管道将读取步骤s+1的数据。这样做可以将步长时间减少到训练的最大值（而不是总和），并减少提取数据所需的时间。
![](media/prefetch执行训练步骤.jpg)

该`tf.data`API提供了`tf.data.Dataset.prefetch`方法。它可用于将产生数据的时间与消耗数据的时间分开。特别是，map使用后台线程和内部缓冲区在请求输入之前，提前从输入数据集中预提取元素。

注意: 要预取的元素数量应等于（或可能大于）单个训练步骤消耗的batch数量。可以手动调整此值，也可以将其设置为`tf.data.experimental.AUTOTUNE`，提示`tf.data`运行时在运行时动态调整值的值。

![](media/prefetch耗时比较.jpg)![](media/prefetch耗时比较2.jpg)

### 2.2 interleave
`tf.data.Dataset.interleave`可以进行并行化数据加载，并交织其他数据集(例如数据文件读取器）的内容。可以通过`cycle_length`参数指定要重叠的数据集数量，而并行度则可以通过`num_parallel_calls`参数指定。

现在使用`interleave`方法的`num_parallel_calls`。这样可以并行加载多个数据集，从而减少了等待文件打开的时间。

![](media/interleave.jpg)![](media/parallel_interleave.jpg)

![](media/interleave_code.jpg)![](media/interleave_code_time.jpg)

### 2.3 map多线程

最简单的方法这里花费在open，read，预处理（map）和训练步骤上的时间加在一起进行一次迭代。
![](media/map.jpg)
![](media/map后.jpg)
![](media/map_code.jpg)
![](media/map_code_time.jpg)


### 2.4 cache方法（用处不大）
`tf.data.Dataset.cache`方法可以将数据集缓存在内存或本地存储。这样可以避免在每个epoch执行某些操作（例如文件打开和数据读取）。

缺点: 以内存换取时间的行为，适合小数据量，**数据量较大请勿使用**!
![](media/cache.jpg)
![](media/cache_code.jpg)
![](media/cache_code_time1.jpg)
![](media/cache_code_time2.jpg)

# 总结
![](media/summary.png)