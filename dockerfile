FROM continuumio/miniconda3

WORKDIR /app

RUN conda install -c conda-forge \
    python=3.10 \
    rdkit \
    pytorch \
    scikit-learn \
    pandas \
    numpy \
    matplotlib \
    seaborn \
    tqdm \
    -y

RUN pip install streamlit

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
