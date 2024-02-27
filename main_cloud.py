import argparse
from queue import Queue
from threading import Thread
import MinkowskiEngine as ME
import torch
import gin


from inference import inference
from src.communication import SplitServer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/semantic_kitti/inference_main.gin")              # train_res16unet34c_prob | train_res16unet34c | train_fpt.gin
    parser.add_argument("--ckpt_path", type=str, default=None)              # train_res16unet34c_prob | train_res16unet34c | train_fpt.gin
    args = parser.parse_args()
    gin.parse_config_file(args.config)

    
    assert torch.cuda.is_available()
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    
    in_queue = Queue()
    out_queue = Queue()

    # Init data and model.
    model = inference(return_model = True, checkpoint_path=args.ckpt_path, data_module_name="SemanticKITTIDataModule")
    server = SplitServer(in_queue, out_queue)

    def run():
        while True:
            data = in_queue.get()
            features = data["batch_features"]
            coordinates = data["batch_coordinates"]
            in_data = ME.TensorField(features=features, coordinates=coordinates, quantization_mode=model.QMODE, device=device)
            logits, emb_mu, emb_sigma = model(in_data)
            out_queue.put({"logits": logits, 
                           "emb_mu_features": emb_mu.features, 
                           "emb_sigma_features": emb_sigma.features,
                           "emb_mu_coordinates": emb_mu.coordinates, 
                           "emb_sigma_coordinates": emb_sigma.coordinates
                           })
            print("Log: number of points:", coordinates.shape)

    t1 = Thread(target=run)
    t2 = Thread(target=server.run, args=("localhost", 8888))

    t1.start()
    t2.start()
