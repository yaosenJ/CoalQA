# 1. 直接使用pytorch原生加载streamlit应用

 ```shell
    git clone https://github.com/yaosenJ/CoalQA.git
    cd CoalQA
    conda create -n CoalQA python=3.10.0 -y
    conda activate CoalQA
    conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install -r requirements.txt
    cd web_app
    streamlit run streamlit_app.py --server.address=127.0.0.1 --server.port 6006
 ```

