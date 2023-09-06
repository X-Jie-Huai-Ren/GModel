
import torch
from model import Generator

class Surgery:
    """
    对旧模型进行离线操作来获得与新环境兼容的新模型
    """
    def __init__(self, model_path, new_model, mode=None) -> None:
        """
        Params:
            model_path: 旧模型参数路径
            new_model: 新模型
            mode: 改变形式
        """
        checkpoints = torch.load(model_path)
        self.old_generetor = checkpoints['generator']
        self.new_generator = new_model.state_dict()

        # for layername, params in generator_params.items():
        #     print(layername)
        #     print(params.shape)

    
    # FC：增加输入特征
    def add_input_feature(self, add_num):
        """
        假设在最后增加
        Params:
            add_num: 增加个数
        """
        






if __name__ == '__main__':

    # new model: add a new feature
    new_model = Generator(z_dim=129, img_dim=784)

    surgery = Surgery(model_path='logs\[09-06]15.55.36\model_0.tar', new_model=new_model)
    
