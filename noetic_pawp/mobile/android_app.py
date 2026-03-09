from __future__ import annotations

import shutil
from pathlib import Path


def generate_android_project(tflite_model_path: str | Path, output_dir: str | Path, package_name: str = "com.noetic.demo") -> Path:
    """Generate a minimal Android project scaffold wired to a TFLite model asset."""
    model = Path(tflite_model_path).expanduser().resolve()
    if not model.exists() or model.suffix != ".tflite":
        raise FileNotFoundError(f"Invalid .tflite model path: {model}")

    out = Path(output_dir).expanduser().resolve()
    app_src = out / "app" / "src" / "main"
    assets = app_src / "assets"
    java_pkg = app_src / "java" / Path(*package_name.split("."))
    res_layout = app_src / "res" / "layout"

    for path in (assets, java_pkg, res_layout):
        path.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(model, assets / model.name)

    main_activity = java_pkg / "MainActivity.java"
    main_activity.write_text(
        """
package PACKAGE_NAME;

import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }
}
""".replace("PACKAGE_NAME", package_name).strip()
        + "\n",
        encoding="utf-8",
    )

    (res_layout / "activity_main.xml").write_text(
        """<LinearLayout xmlns:android=\"http://schemas.android.com/apk/res/android\"
    android:layout_width=\"match_parent\" android:layout_height=\"match_parent\"
    android:gravity=\"center\" android:orientation=\"vertical\">
    <TextView android:layout_width=\"wrap_content\" android:layout_height=\"wrap_content\"
        android:text=\"Noetic Android TFLite Demo\" />
</LinearLayout>
""",
        encoding="utf-8",
    )
    (out / "README.md").write_text("Import this folder in Android Studio as an existing project scaffold.\n", encoding="utf-8")
    return out
