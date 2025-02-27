import mteb
from sentence_transformers import SentenceTransformer

model_name = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_name)

benchmark = mteb.get_benchmark("MTEB(eng, v1)")
evaluation = mteb.MTEB(tasks=benchmark)
results = evaluation.run(model, output_folder=f"results/{model_name}")
