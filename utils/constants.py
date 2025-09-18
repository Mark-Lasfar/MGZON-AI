MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-oss-120b:cerebras")
SECONDARY_MODEL_NAME = os.getenv("SECONDARY_MODEL_NAME", "mistralai/Mixtral-8x7B-Instruct-v0.1")
TERTIARY_MODEL_NAME = os.getenv("TERTIARY_MODEL_NAME", "llama/Llama-3.1-8B-Instruct:featherless-ai")
CLIP_BASE_MODEL = os.getenv("CLIP_BASE_MODEL", "Salesforce/blip-image-captioning-large")
CLIP_LARGE_MODEL = os.getenv("CLIP_LARGE_MODEL", "openai/clip-vit-large-patch14")
ASR_MODEL = os.getenv("ASR_MODEL", "openai/whisper-large-v3")
TTS_MODEL = os.getenv("TTS_MODEL", "facebook/mms-tts-ara")
IMAGE_GEN_MODEL = os.getenv("IMAGE_GEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct:novita")
SECONDARY_IMAGE_GEN_MODEL = os.getenv("SECONDARY_IMAGE_GEN_MODEL", "black-forest-labs/FLUX.1-dev")

MODEL_ALIASES = {
    "advanced": MODEL_NAME,
    "standard": SECONDARY_MODEL_NAME,
    "light": TERTIARY_MODEL_NAME,
    "image_base": CLIP_BASE_MODEL,
    "image_advanced": CLIP_LARGE_MODEL,
    "audio": ASR_MODEL,
    "tts": TTS_MODEL,
    "image_gen": IMAGE_GEN_MODEL,
    "secondary_image_gen": SECONDARY_IMAGE_GEN_MODEL
}