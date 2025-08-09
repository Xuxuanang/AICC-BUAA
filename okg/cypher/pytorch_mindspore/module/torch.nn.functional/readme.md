## 1 torch.nn.functional

**torch.nn.functional**中的函数和**torch.nn**中的函数都提供了常用的神经网络操作，包括激活函数、损失函数、池化操作等。它们的主要区别如下：

1. **函数形式 vs. 类形式**

   torch.nn.functional中的函数是以函数形式存在的，而torch.nn中的函数是以类形式存在的。torch.nn.functional中的函数是纯函数，没有与之相关联的可学习参数。而torch.nn中的函数是torch.nn.Module的子类，可以包含可学习参数，并且可以在模型中作为子模块使用。

   torch.nn.ReLU如下所示：

   ```python
   class ReLU(Module):
       __constants__ = ['inplace']
       inplace: bool
   
       def __init__(self, inplace: bool = False):
           super(ReLU, self).__init__()
           self.inplace = inplace
   
       def forward(self, input: Tensor) -> Tensor:
           return F.relu(input, inplace=self.inplace)
   
       def extra_repr(self) -> str:
           inplace_str = 'inplace=True' if self.inplace else ''
           return inplace_str
   ```

   torch.nn.functional.relu如下所示：

   ```python
   def relu(input: Tensor, inplace: bool = False) -> Tensor:
       r"""relu(input, inplace=False) -> Tensor
   
       Applies the rectified linear unit function element-wise. See
       :class:`~torch.nn.ReLU` for more details.
       """
       if has_torch_function_unary(input):
           return handle_torch_function(relu, (input,), input, inplace=inplace)
       if inplace:
           result = torch.relu_(input)
       else:
           result = torch.relu(input)
       return result
   ```

2. **参数传递方式**

   torch.nn.functional中的函数是直接传递张量作为参数的，而torch.nn中的函数需要实例化后，将张量作为实例的调用参数。

3. **状态管理**

   由于torch.nn.functional中的函数是纯函数，没有与之相关联的参数或状态，因此无法直接管理和访问函数的内部状态。

   而torch.nn中的函数是torch.nn.Module的子类，可以管理和访问模块的内部参数和状态。

4. 使用示例

   



参考：https://blog.csdn.net/m0_70484757/article/details/131353444





## 2 mindspore.ops

提问：在看MindSpore过程中发现nn和ops里都有Conv1d/2d/3d，看描述是一致的，求助nn和ops里的同一个算子有什么差别，该用哪个？

![image-20231130105028691](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20231130105028691.png)

![image-20231130105044129](C:\Users\lyz\AppData\Roaming\Typora\typora-user-images\image-20231130105044129.png)



参考：https://www.zhihu.com/question/538954604