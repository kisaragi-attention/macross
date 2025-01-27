import torch
print('CUDA版本:',torch.version.cuda)
print('Pytorch版本:',torch.__version__)
print('显卡是否可用:','可用' if(torch.cuda.is_available()) else '不可用')
print('显卡数量:',torch.cuda.device_count())
print('是否支持BF16数字格式:','支持' if (torch.cuda.is_bf16_supported()) else '不支持')
print('当前显卡型号:',torch.cuda.get_device_name())
print('当前显卡的CUDA算力:',torch.cuda.get_device_capability())
print('当前显卡的总显存:',torch.cuda.get_device_properties(0).total_memory/1024/1024/1024,'GB')
print('是否支持TensorCore:','支持' if (torch.cuda.get_device_properties(0).major >= 7) else '不支持')
print('当前显卡的显存使用率:',torch.cuda.memory_allocated(0)/torch.cuda.get_device_properties(0).total_memory*100,'%')
 
  
# Check if CUDA is available
is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    print("CUDA is available.")
else:
    print("CUDA is not available.")
               
# Check if cuDNN is available
from torch.backends import cudnn
cudnn.is_acceptable_type = lambda t: t.is_floating_point or t.is_integer
is_cudnn_available = cudnn.is_available()
if is_cudnn_available:
    print("cuDNN is available.")
else:
    print("cuDNN is not available.")
