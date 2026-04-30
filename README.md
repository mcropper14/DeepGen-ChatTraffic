# DeepGen-ChatTraffic

The orginal repository is listed here: [ChatTraffic Repository](https://github.com/ChyaZhang/ChatTraffic)

If you get an error about PyTorch lightning try this: ``` pip install packaging==20.9 torchmetrics==0.6.0 ``` 

This to fix the diffusion model error: ``` pip install "kornia<0.7" ```

Data set: [chatTraffic](https://drive.google.com/file/d/1uTLiB5-WnfX46PizrnQ8diUCZScZQuSl/view?usp=sharing)

Training plots [plots](https://drive.google.com/drive/u/1/folders/1gdQa4xsLskdbwLzw-eEMvRDNMisXal8U) 

Sample outputs  [samples](https://drive.google.com/drive/folders/187_QWBnjrfnl_LaRfgcETFuvWPyGMO6S?usp=drive_link)
the output is an interactive map 

Our flow matching implementation can be found at: /ldm/models/flow_matching.py 

The evaluation can be found at: scripts/evaluate.py

The temporal data generation can be found at: scripts/sequence_generation.py

Training can be run via: ``` CUDA_VISIBLE_DEVICES=<GPU_ID> python main.py --base configs/latent-diffusion/traffic.yaml -t --gpus 0, ```
