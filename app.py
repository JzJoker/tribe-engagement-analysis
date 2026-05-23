from tribev2 import TribeModel

model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder="./cache",
    config_update={
        "data.features_to_use": ["audio", "video"],
        "data.audio_feature.device": "cpu",
        "data.video_feature.image.device": "cpu",
    },
)

df = model.get_events_dataframe(video_path="tiktok_products.mp4")
preds, segments = model.predict(events=df)
print(preds.shape)  # (n_timesteps, n_vertices)
