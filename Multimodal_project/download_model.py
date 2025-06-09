from transformers import AutoProcessor, AutoModelForImageTextToText
    
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", cache_dir ="/cluster/work/sachan/shridhar/models/cache", trust_remote_code=True, token="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")
# tokenizer.save_pretrained("/cluster/work/sachan/shridhar/models/gemma2-2b-it", from_pt=True)

# model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", cache_dir ="/cluster/work/sachan/shridhar/models/cache", trust_remote_code=True, token="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")
# model.save_pretrained("/cluster/work/sachan/shridhar/models/gemma2-2b-it", from_pt=True)

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", cache_dir ="/cluster/work/sachan/piyushi/models/cache", trust_remote_code=True)
# model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", cache_dir ="/cluster/work/sachan/piyushi/models/cache", trust_remote_code=True)


# processor = AutoProcessor.from_pretrained("facebook/chameleon-7b", trust_remote_code=True)
# model = AutoModelForImageTextToText.from_pretrained("facebook/chameleon-7b", trust_remote_code=True)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

processor.save_pretrained("/cluster/home/pgoyal/qwen-vl-7B")
model.save_pretrained("/cluster/home/pgoyal/qwen-vl-7B")

# from huggingface_hub import snapshot_download

# snapshot_download(repo_id="facebook/chameleon-7b", repo_type="model", local_dir="chameleon-7b",)