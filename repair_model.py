import zipfile
import json
import shutil
import os

input_model = os.path.join("Artifacts", "BiGRU_Model.keras")
output_model = os.path.join("Artifacts", "BiGRU_Model_fixed.keras")

print("Reading:", input_model)

with zipfile.ZipFile(input_model, "r") as zin:
    with zipfile.ZipFile(output_model, "w", zipfile.ZIP_DEFLATED) as zout:

        for item in zin.infolist():
            data = zin.read(item.filename)

            if item.filename == "config.json":
                print("Found config.json")

                config = json.loads(data)

                def clean_config(obj):
                    if isinstance(obj, dict):
                        obj.pop("quantization_config", None)

                        for value in obj.values():
                            clean_config(value)

                    elif isinstance(obj, list):
                        for value in obj:
                            clean_config(value)

                clean_config(config)

                data = json.dumps(config).encode("utf-8")

                print("Removed quantization_config entries.")

            zout.writestr(item, data)

print()
print("Fixed model created:")
print(output_model)