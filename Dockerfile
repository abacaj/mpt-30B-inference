FROM python:3.10

RUN apt-get update && apt-get install -y \
python3 \
python3-pip \
git \
curl
RUN pip install --upgrade pip
RUN git clone https://github.com/abacaj/mpt-30B-inference.git
RUN cd mpt-30B-inference && pip install -r requirements.txt
RUN python download_model.py && python inference.py
ENTRYPOINT ["tail", "-f", "/dev/null"]

# Usage: docker build -t img_name . && docker run --name=ct_mpt-30b img_name
