from transformers import AutoProcessor, AutoModelForImageTextToText
    
# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", cache_dir ="/cluster/work/sachan/shridhar/models/cache", trust_remote_code=True, token="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")
# tokenizer.save_pretrained("/cluster/work/sachan/shridhar/models/gemma2-2b-it", from_pt=True)

# model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", cache_dir ="/cluster/work/sachan/shridhar/models/cache", trust_remote_code=True, token="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")
# model.save_pretrained("/cluster/work/sachan/shridhar/models/gemma2-2b-it", from_pt=True)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", cache_dir ="/cluster/work/sachan/piyushi/models/cache", trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", cache_dir ="/cluster/work/sachan/piyushi/models/cache", trust_remote_code=True)
processor.save_pretrained("/cluster/work/sachan/piyushi/models/qwen-vl-3B-it", from_pt=True)
model.save_pretrained("/cluster/work/sachan/piyushi/models/qwen-vl-3B-it", from_pt=True)