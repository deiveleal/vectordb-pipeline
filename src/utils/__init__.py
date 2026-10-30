"""Utilitarios compartilhados pelas camadas do pipeline.

    oai_connect        cliente OAI-PMH do arXiv, com paginacao, pausa e retry
    storage_functions  leitura e escrita no S3/RustFS
    load_boto_client   construcao do cliente boto3 e endereco do RustFS
    load_catalog       catalogo Iceberg (SQLite) das tabelas silver e gold
    load_env           carga do .env da raiz
    load_logger        logging estruturado, em Brasilia e com offset
    load_time          fuso e data de negocio do pipeline

Nenhum destes modulos importa uma camada -- a dependencia so corre no sentido
camada -> utilitario. O load_logger depende do load_time (pelo fuso), e essa e
a unica relacao entre eles.
"""
