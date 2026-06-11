# Pipeline de asistente de voz

Pipeline completo: captura -> VAD -> ASR -> LLM -> TTS:

```bash
cd code
python3 main.py
```

Ejecuta los tests:

```bash
python3 -m unittest discover -s tests -v
```