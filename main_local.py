import argparse
from src.communication import SplitClient
import gin
from inference import inference

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/semantic_kitti/inference_main.gin")              # train_res16unet34c_prob | train_res16unet34c | train_fpt.gin
    parser.add_argument("--ckpt_path", type=str, default=None)              # train_res16unet34c_prob | train_res16unet34c | train_fpt.gin
    args = parser.parse_args()
    gin.parse_config_file(args.config)
    
    client = SplitClient('http://localhost:8888', 'secret_api_key')
    
    inference(checkpoint_path=args.ckpt_path, data_module_name="SemanticKITTIDataModule", client = client, number_inputs=10)

