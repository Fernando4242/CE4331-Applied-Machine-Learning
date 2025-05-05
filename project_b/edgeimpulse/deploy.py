import edgeimpulse as ei
import tensorflow as tf

# The key here.
ei.API_KEY = "ei_...."
model_path = "./out/model_quantized_int8_io.tflite"

profile = ei.model.profile(model=model_path, device='espressif-esp32');
print(profile.summary())

ei.model.deploy(model=model_path,
                model_input_type=ei.model.input_type.OtherInput(),
                model_output_type=ei.model.output_type.Classification(),
                output_directory=".")