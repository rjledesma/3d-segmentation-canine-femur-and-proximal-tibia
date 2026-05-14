from typing import Dict, Any
import traceback
import zipfile

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from app.services.dicom_service import prepare_zip_as_nnunet_input
from app.services.crop_service import crop_ct_for_femur_tibia_inference
from app.services.file_service import (
    create_case_id,
    create_case_dirs,
    save_upload_file,
    normalize_nnunet_input_name,
    PROCESSED_DIR,
)
from app.services.predict_service import run_nnunet_prediction
from app.services.mesh_service import convert_prediction_to_stls

app = FastAPI(title="Femur-Tibia 3D Segmentation Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/processed", StaticFiles(directory=str(PROCESSED_DIR)), name="processed")


JOBS: Dict[str, Dict[str, Any]] = {}


def is_supported_file(filename: str) -> bool:
    lower = filename.lower()
    return lower.endswith(".nii") or lower.endswith(".nii.gz") or lower.endswith(".zip")


def is_zip_file(filename: str) -> bool:
    return filename.lower().endswith(".zip")


def update_job(case_id: str, **kwargs):
    if case_id in JOBS:
        JOBS[case_id].update(kwargs)


def prepare_uploaded_input(case_id: str, filename: str, input_path):
    """
    If the uploaded file is a NIfTI file, it is already saved as input_0000.nii.gz or input_0000.nii.
    If the uploaded file is a DICOM ZIP, extract and convert it to input_0000.nii.gz.
    Then crop large CT volumes before nnU-Net inference.
    """
    dirs = create_case_dirs(case_id)

    if is_zip_file(filename):
        update_job(
            case_id,
            status="converting_input",
            progress=20,
            message="DICOM ZIP uploaded. Converting DICOM series to NIfTI...",
        )

        print(f"[DICOM] Converting ZIP to NIfTI for {case_id}")

        converted_nifti = prepare_zip_as_nnunet_input(
            zip_path=input_path,
            case_dir=dirs["case_dir"],
            input_dir=dirs["input_dir"],
        )

        # Keep a backup of the full converted CT OUTSIDE the nnU-Net input folder.
        # Important: do not place any *_0000.nii.gz backup inside input_dir,
        # because nnU-Net will treat it as another prediction case.
        full_backup_dir = dirs["case_dir"] / "full_input_backup"
        full_backup_dir.mkdir(parents=True, exist_ok=True)

        full_backup = full_backup_dir / "input_full.nii.gz"

        if converted_nifti.exists() and not full_backup.exists():
            converted_nifti.replace(full_backup)

        update_job(
            case_id,
            status="cropping_input",
            progress=28,
            message="Cropping CT to femur/proximal tibia region...",
        )

        crop_ct_for_femur_tibia_inference(
            input_nifti_path=full_backup,
            output_nifti_path=dirs["input_dir"] / "input_0000.nii.gz",
            bone_threshold=250,
            margin_xy=48,
            margin_long=32,
            max_long_axis_voxels=900,
        )

    else:
        # Optional: crop only very large NIfTI uploads.
        # For already cropped Dataset002 images, this will usually be skipped or small.
        pass


def process_segmentation_job(case_id: str):
    """
    Runs in background:
    1. Optional DICOM ZIP to NIfTI conversion already handled before this job starts
    2. nnU-Net prediction
    3. STL conversion
    4. saves result URLs to JOBS
    """
    try:
        dirs = create_case_dirs(case_id)

        update_job(
            case_id,
            status="predicting",
            progress=35,
            message="Running nnU-Net inference...",
        )

        print(f"[JOB {case_id}] Running nnU-Net prediction...")
        prediction_path = run_nnunet_prediction(
            input_dir=dirs["input_dir"],
            output_dir=dirs["output_dir"],
        )

        update_job(
            case_id,
            status="converting",
            progress=80,
            message="Converting prediction mask to STL meshes...",
            prediction_mask_url=f"/processed/{case_id}/output/{prediction_path.name}",
        )

        print(f"[JOB {case_id}] Converting prediction to STL...")
        stl_paths = convert_prediction_to_stls(
            prediction_path=prediction_path,
            mesh_dir=dirs["mesh_dir"],
        )

        JOBS[case_id] = {
            "case_id": case_id,
            "status": "complete",
            "progress": 100,
            "message": "Segmentation complete.",
            "prediction_mask_url": f"/processed/{case_id}/output/{prediction_path.name}",
            "femur_stl_url": f"/processed/{case_id}/meshes/{stl_paths['femur_stl'].name}",
            "proximal_tibia_stl_url": f"/processed/{case_id}/meshes/{stl_paths['proximal_tibia_stl'].name}",
            "labels": {
                "1": "femur",
                "2": "proximal_tibia",
            },
        }

        print(f"[JOB {case_id}] Complete.")

    except Exception as e:
        error_text = str(e)
        traceback_text = traceback.format_exc()

        print(f"[JOB {case_id}] ERROR:", error_text)
        print(traceback_text)

        update_job(
            case_id,
            status="error",
            progress=100,
            message="Segmentation failed.",
            error=error_text,
        )


@app.get("/")
def root():
    return {
        "message": "Femur-Tibia 3D Segmentation Backend is running",
        "model": "Dataset002_HumanFemurProximalTibia",
        "supported_inputs": [".nii", ".nii.gz", ".zip DICOM"],
        "labels": {
            "1": "femur",
            "2": "proximal_tibia",
        },
    }


@app.post("/segment/start")
async def start_segmentation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    filename = file.filename or "input.nii.gz"

    if not is_supported_file(filename):
        raise HTTPException(
            status_code=400,
            detail="Only .nii, .nii.gz, and .zip DICOM files are supported.",
        )

    try:
        case_id = create_case_id()
        dirs = create_case_dirs(case_id)

        input_path = normalize_nnunet_input_name(filename, dirs["input_dir"])
        save_upload_file(file, input_path)

        JOBS[case_id] = {
            "case_id": case_id,
            "status": "queued",
            "progress": 10,
            "message": "File uploaded. Job queued.",
            "filename": filename,
        }

        print(f"[UPLOAD] {filename} -> {input_path}")

        if is_zip_file(filename):
            prepare_uploaded_input(case_id, filename, input_path)

        print(f"[JOB {case_id}] Queued.")
        background_tasks.add_task(process_segmentation_job, case_id)

        return {
            "case_id": case_id,
            "status": JOBS[case_id]["status"],
            "progress": JOBS[case_id]["progress"],
            "message": "File uploaded. Segmentation started.",
        }

    except Exception as e:
        JOBS[case_id] = {
            "case_id": case_id,
            "status": "error",
            "progress": 100,
            "message": "Upload or input conversion failed.",
            "error": str(e),
            "filename": filename,
        }
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/segment/status/{case_id}")
def get_segmentation_status(case_id: str):
    if case_id not in JOBS:
        raise HTTPException(status_code=404, detail="Case ID not found.")

    return JOBS[case_id]


@app.post("/segment")
async def segment(file: UploadFile = File(...)):
    """
    Synchronous endpoint kept for compatibility/testing.
    Frontend should use /segment/start instead.
    """
    filename = file.filename or "input.nii.gz"

    if not is_supported_file(filename):
        raise HTTPException(
            status_code=400,
            detail="Only .nii, .nii.gz, and .zip DICOM files are supported.",
        )

    try:
        case_id = create_case_id()
        dirs = create_case_dirs(case_id)

        input_path = normalize_nnunet_input_name(filename, dirs["input_dir"])
        save_upload_file(file, input_path)

        print(f"[UPLOAD] {filename} -> {input_path}")

        JOBS[case_id] = {
            "case_id": case_id,
            "status": "queued",
            "progress": 10,
            "message": "File uploaded.",
            "filename": filename,
        }

        if is_zip_file(filename):
            prepare_uploaded_input(case_id, filename, input_path)

        prediction_path = run_nnunet_prediction(
            input_dir=dirs["input_dir"],
            output_dir=dirs["output_dir"],
        )

        stl_paths = convert_prediction_to_stls(
            prediction_path=prediction_path,
            mesh_dir=dirs["mesh_dir"],
        )

        return JSONResponse(
            {
                "case_id": case_id,
                "status": "success",
                "message": "Segmentation complete.",
                "prediction_mask_url": f"/processed/{case_id}/output/{prediction_path.name}",
                "femur_stl_url": f"/processed/{case_id}/meshes/{stl_paths['femur_stl'].name}",
                "proximal_tibia_stl_url": f"/processed/{case_id}/meshes/{stl_paths['proximal_tibia_stl'].name}",
                "labels": {
                    "1": "femur",
                    "2": "proximal_tibia",
                },
            }
        )

    except HTTPException:
        raise

    except Exception as e:
        print("[ERROR]", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{case_id}")
def download_case_results(case_id: str):
    case_dir = PROCESSED_DIR / case_id
    output_dir = case_dir / "output"
    mesh_dir = case_dir / "meshes"

    prediction_mask = output_dir / "input.nii.gz"
    femur_stl = mesh_dir / "femur.stl"
    tibia_stl = mesh_dir / "proximal_tibia.stl"

    if not case_dir.exists():
        raise HTTPException(status_code=404, detail=f"Case folder not found: {case_id}")

    if not prediction_mask.exists():
        raise HTTPException(status_code=404, detail="Prediction mask not found.")

    if not femur_stl.exists():
        raise HTTPException(status_code=404, detail="Femur STL not found.")

    if not tibia_stl.exists():
        raise HTTPException(status_code=404, detail="Proximal tibia STL not found.")

    zip_path = case_dir / f"{case_id}_results.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(prediction_mask, arcname="prediction_mask.nii.gz")
        z.write(femur_stl, arcname="femur.stl")
        z.write(tibia_stl, arcname="proximal_tibia.stl")

    return FileResponse(
        path=str(zip_path),
        filename=f"{case_id}_results.zip",
        media_type="application/zip",
    )
