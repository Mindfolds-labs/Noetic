# Noetic TensorFlow Integration

## Visão arquitetural
Módulos em `noetic_pawp.data`, `noetic_pawp.export`, `noetic_pawp.web` e `noetic_pawp.mobile` ampliam o core sem reescrever componentes cognitivos.

## Fluxo TF Data
Use `create_nyuv2_dataset`, `create_kitti_dataset` ou `create_multimodal_dataset` com estrutura `<root>/<split>/*.jpg|*.png`.

## RIVE em TensorFlow
`RIVEPreprocessor` produz mapas `radial`, `frustum` e vetor `rive_coeffs` para consumo multimodal.

## Exportação ONNX
`export_pawp_tokenizer_to_onnx` e `export_noetic_bridge_to_onnx` validam o grafo com `onnx.checker`.

## Demo web/mobile
- Web: `generate_tfjs_demo(saved_model_dir, output_dir)`
- Android: `generate_android_project(model.tflite, output_dir)`

## Limitações e roadmap
- Conversão TF.js depende de `tensorflowjs_converter` instalado.
- Benchmarks atuais são baseline e podem evoluir para perfis por dispositivo.
