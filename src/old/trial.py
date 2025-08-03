import onnxruntime as ort

print("Available providers:", ort.get_available_providers())

session = ort.InferenceSession("your_model.onnx")  # Replace with your actual model path
print("Session providers:", session.get_providers())
print("Using:", session.get_provider_options())
