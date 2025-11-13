import numpy as np
from pathlib import Path
from validation_tool.universal_opnav_tester import (
    UniversalOpNavTester, TestImage, CRAResult, TargetBody
)

def setup_tester():
    """Initialize the testing framework"""
    
    tester = UniversalOpNavTester(
        output_dir=Path("./found_cra_results"),
        moon_radius_km=1737.4,
        earth_radii_km=(6378.137, 6378.137, 6356.752)  # WGS84
    )
    
    def custom_moon_error_model(range_km):
        """Expected error model"""
        return {
            'radius_1sigma_px': 0.08,  # Expected performance
            'centroid_1sigma_px': 0.12,
            'terrain_bias_max_px': 0.25
        }
    
    tester.set_custom_error_model(TargetBody.MOON, custom_moon_error_model)
    
    return tester

def create_test_image_from_data(
    image_array: np.ndarray,  # Grayscale image
    
    # Ground truth (BET)
    position_camera_m: np.ndarray,  # [x, y, z] in meters, camera frame
    velocity_camera_ms: np.ndarray,  # [vx, vy, vz] in m/s
    sun_direction_camera: np.ndarray,  # [x, y, z] unit vector
    
    # Camera calibration
    camera_K: np.ndarray,  # 3x3 calibration matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    distortion_params: dict,  # {'k1': ..., 'k2': ..., 'k3': ..., 'p1': ..., 'p2': ...}
    
    # Metadata
    timestamp: float,
    image_id: str,
    pass_id: str,
    target: TargetBody = TargetBody.MOON,
    
    # Optional
    bet_uncertainty_range_m: float = None,
    bet_uncertainty_pointing_arcsec: float = None
) -> TestImage:
    
    # Calculate derived values
    range_m = np.linalg.norm(position_camera_m)
    
    # Target body parameters
    if target == TargetBody.MOON:
        body_radii = np.array([1737400, 1737400, 1737400])  # meters
        body_orientation = np.eye(3)  # Assume aligned
    else:  # Earth
        body_radii = np.array([6378137, 6378137, 6356752])  # WGS84 in meters
        body_orientation = np.eye(3)
    
    # Attitude (for simple case, assume camera aligned with inertial)
    attitude = np.eye(3)
    
    # Sun direction in inertial frame (for simple case, same as camera)
    sun_direction_inertial = sun_direction_camera.copy()
    
    # Phase angle (sun-target-observer angle)
    obs_to_target = position_camera_m / range_m
    phase_angle_rad = np.arccos(np.dot(-obs_to_target, sun_direction_camera))
    phase_angle_deg = np.rad2deg(phase_angle_rad)
    
    return TestImage(
        image=image_array,
        position_camera=position_camera_m,
        velocity_camera=velocity_camera_ms,
        attitude_camera_to_inertial=attitude,
        sun_direction_camera=sun_direction_camera,
        sun_direction_inertial=sun_direction_inertial,
        body_radii=body_radii,
        body_orientation=body_orientation,
        target_body=target,
        camera_K=camera_K,
        distortion=distortion_params,
        timestamp=timestamp,
        image_id=image_id,
        pass_id=pass_id,
        range_m=range_m,
        phase_angle_deg=phase_angle_deg,
        bet_uncertainty_range_m=bet_uncertainty_range_m,
        bet_uncertainty_pointing_arcsec=bet_uncertainty_pointing_arcsec
    )


def run_found_cra(test_image: TestImage) -> CRAResult:
    """
    CALL FOUND CRA IMPLEMENTATION

    """
    
    return CRAResult(
        position_camera=position_estimate,
        velocity_camera=None,
        detected_limb_points=detected_limb,
        apparent_radius_pixels=0.0, 
        centroid_pixels=None,
        success=True,
        converged=True,
        n_iterations=0,
        processing_time_s=elapsed,
        algorithm_type="non-iterative"
    )

def load_test_dataset():
    """
    Load test images and ground truth
    
    REQUIRED DATA FOR EACH IMAGE:
    1. Image array (grayscale)
    2. Camera calibration (K matrix, distortion)
    3. Ground truth position (meters, camera frame)
    4. Ground truth velocity (m/s, camera frame)
    5. Sun direction (unit vector, camera frame)
    6. Timestamp
    7. Pass ID and image ID
    """
    
    test_images = []
    from synthetic_data_generator import SyntheticMoonImageGenerator, PassGenerator
    
    generator = SyntheticMoonImageGenerator(
        image_size=(512, 512),
        focal_length=2000.0
    )
    
    pass_gen = PassGenerator(generator)
    
    # Generate a certification pass
    cert_pass = pass_gen.generate_pass(
        range_km_start=220000,
        range_km_end=136000,
        n_images=100,  # Full 2-hour pass at 30 sec intervals
        phase_angle_deg=75,
        pass_id="CertPass"
    )
    
    # Camera parameters
    camera_K = np.array([[2000, 0, 256],
                        [0, 2000, 256],
                        [0, 0, 1]])
    distortion = {'k1': 0, 'k2': 0, 'k3': 0, 'p1': 0, 'p2': 0}
    
    for test_case, metadata in cert_pass:
        test_img = create_test_image_from_data(
            image_array=test_case.image,
            position_camera_m=test_case.position_camera,
            velocity_camera_ms=np.array([0, 0, 0]),  # Static for this example
            sun_direction_camera=test_case.sun_direction_camera,
            camera_K=camera_K,
            distortion_params=distortion,
            timestamp=metadata['timestamp'],
            image_id=metadata['image_id'],
            pass_id=metadata['pass_id'],
            target=TargetBody.MOON,
            bet_uncertainty_range_m=7300,  # 24kft from Artemis I
            bet_uncertainty_pointing_arcsec=36.2  # From Artemis I
        )
        test_images.append(test_img)
    
    """
    from data_loader import load_pds_image, load_spice_data
    
    for image_file in image_list:
        # Load image
        image = load_pds_image(image_file)
        
        # Get ground truth from SPICE
        spice_data = load_spice_data(image_file.timestamp)
        
        test_img = create_test_image_from_data(
            image_array=image.data,
            position_camera_m=spice_data.position,
            velocity_camera_ms=spice_data.velocity,
            sun_direction_camera=spice_data.sun_direction,
            camera_K=image.camera_K,
            distortion_params=image.distortion,
            timestamp=image.timestamp,
            image_id=image.id,
            pass_id=image.pass_id,
            target=TargetBody.MOON
        )
        test_images.append(test_img)
    """
    
    # Add more passes at different ranges (like Artemis I Table 1)
    pass_configs = [
        (89000, 10, "Pass5"),
        (51000, 10, "Pass6"),
        (19000, 12, "Pass7"),
        (43000, 10, "Pass9"),
        (147500, 10, "Pass22"),
    ]
    
    for range_km, n_imgs, pass_id in pass_configs:
        pass_data = pass_gen.generate_pass(
            range_km_start=range_km * 1.05,
            range_km_end=range_km * 0.95,
            n_images=n_imgs,
            phase_angle_deg=np.random.uniform(60, 100),
            pass_id=pass_id
        )
        
        for test_case, metadata in pass_data:
            test_img = create_test_image_from_data(
                image_array=test_case.image,
                position_camera_m=test_case.position_camera,
                velocity_camera_ms=np.array([0, 0, 0]),
                sun_direction_camera=test_case.sun_direction_camera,
                camera_K=camera_K,
                distortion_params=distortion,
                timestamp=metadata['timestamp'],
                image_id=metadata['image_id'],
                pass_id=metadata['pass_id'],
                target=TargetBody.MOON
            )
            test_images.append(test_img)
    
    return test_images

def main():
    print("="*80)
    print("FOUND CRA ALGORITHM VALIDATION")
    print("Recreating Artemis I OpNav Performance Analysis")
    print("="*80)
    
    # Initialize tester
    print("\n[1/5] Initializing tester...")
    tester = setup_tester()
    
    # Load test data
    print("\n[2/5] Loading test dataset...")
    test_images = load_test_dataset()
    print(f"  Loaded {len(test_images)} test images")
    
    # Process all images with CRA
    print("\n[3/5] Running FOUND CRA on all images...")
    n_success = 0
    n_failed = 0
    
    for i, test_img in enumerate(test_images):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i+1}/{len(test_images)}...")
        
        try:
            # Run algorithm
            cra_result = run_found_cra(test_img)
            
            if cra_result.success:
                # Add to tester
                tester.add_test_case(test_img, cra_result)
                n_success += 1
            else:
                n_failed += 1
                
        except Exception as e:
            print(f"  Error on {test_img.image_id}: {e}")
            n_failed += 1
    
    print(f"\n  Results: {n_success} succeeded, {n_failed} failed")
    
    # Generate all figures
    print("\n[4/5] Generating performance plots...")
    
    # Figure 7: Moon radius residuals vs range (all passes)
    print("  Creating Figure 7...")
    tester.plot_figure_7(
        target=TargetBody.MOON,
        filename="figure_7_moon_radius_all_passes.png"
    )
    
    # Figure 8: Moon centroid residuals in illumination frame
    print("  Creating Figure 8...")
    tester.plot_figure_8(
        target=TargetBody.MOON,
        filename="figure_8_moon_centroid_all_passes.png"
    )
    
    # Figures 9-10: Certification pass details
    print("  Creating Figures 9-10...")
    tester.plot_certification_pass(
        pass_id="CertPass",
        target=TargetBody.MOON,
        filename="figures_9_10_cert_pass.png"
    )
    
    # For iterative results, create comparison (Figures 13-15)
    # print("  Creating Figures 13-15...")
    # tester.plot_algorithm_comparison(
    #     pass_id="CertPass",
    #     results_iterative=iterative_results,
    #     filename="figures_13_15_algorithm_comparison.png"
    # )
    
    # For Earth images
    # print("  Creating Figure 16 (Earth radius)...")
    # tester.plot_figure_7(
    #     target=TargetBody.EARTH,
    #     filename="figure_16_earth_radius_all_passes.png"
    # )
    
    # Statistics
    print("\n[5/5] Generating statistics...")
    
    # Overall Moon stats
    tester.print_statistics_table(target=TargetBody.MOON)
    
    # Certification pass stats (like Table 2 in paper)
    tester.print_statistics_table(
        target=TargetBody.MOON,
        pass_id="CertPass"
    )
    
    # Save statistics to file
    stats = tester.generate_statistics_table(target=TargetBody.MOON, pass_id="CertPass")
    stats_file = tester.output_dir / "statistics_table2.txt"
    with open(stats_file, 'w') as f:
        f.write("FOUND CRA Performance Statistics (Table 2 format)\n")
        f.write("="*70 + "\n\n")
        f.write(f"Moon Certification Pass - {stats['n_images']} images\n\n")
        f.write(f"{'Parameter':<20} {'Mean':>12} {'Std Dev':>12}\n")
        f.write("-"*46 + "\n")
        f.write(f"{'Radius (pixels)':<20} {stats['radius']['mean']:>12.4f} {stats['radius']['std']:>12.4f}\n")
        f.write(f"{'Centroid X (pixels)':<20} {stats['centroid_x']['mean']:>12.4f} {stats['centroid_x']['std']:>12.4f}\n")
        f.write(f"{'Centroid Y (pixels)':<20} {stats['centroid_y']['mean']:>12.4f} {stats['centroid_y']['std']:>12.4f}\n")
    
    print(f"\n  Statistics saved to {stats_file}")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {tester.output_dir}")
    print("\nGenerated files:")
    print("  • figure_7_moon_radius_all_passes.png")
    print("  • figure_8_moon_centroid_all_passes.png")
    print("  • figures_9_10_cert_pass.png")
    print("  • statistics_table2.txt")
    print("="*80 + "\n")

def test_single_image():
    """Test CRA on a single image first"""
    
    print("Testing single image...")
    
    # Generate one test image
    from synthetic_data_generator import SyntheticMoonImageGenerator
    
    generator = SyntheticMoonImageGenerator()
    test_case = generator.generate(
        range_km=150000,
        phase_angle_deg=90
    )
    
    # Convert to TestImage format
    camera_K = np.array([[2000, 0, 256],
                        [0, 2000, 256],
                        [0, 0, 1]])
    
    test_img = create_test_image_from_data(
        image_array=test_case.image,
        position_camera_m=test_case.position_camera,
        velocity_camera_ms=np.array([0, 0, 0]),
        sun_direction_camera=test_case.sun_direction_camera,
        camera_K=camera_K,
        distortion_params={'k1': 0, 'k2': 0, 'k3': 0, 'p1': 0, 'p2': 0},
        timestamp=0.0,
        image_id="test_001",
        pass_id="SingleTest",
        target=TargetBody.MOON
    )
    
    # Run CRA
    result = run_found_cra(test_img)
    
    # Quick validation
    tester = setup_tester()
    tester.add_test_case(test_img, result)
    
    residuals = tester.residuals[0]
    
    print(f"\nResults:")
    print(f"  Range error: {residuals.range_error_km:.2f} km")
    print(f"  Radius residual: {residuals.radius_residual_pixels:.4f} pixels")
    print(f"  Centroid X: {residuals.centroid_illum_x_pixels:.4f} pixels")
    print(f"  Centroid Y: {residuals.centroid_illum_y_pixels:.4f} pixels")
    print(f"  Bearing error: {residuals.bearing_error_arcsec:.2f} arcsec")
    
    if abs(residuals.radius_residual_pixels) < 0.5:
        print("\n✓ Good! Radius residual < 0.5 pixels")
    else:
        print("\n✗ Warning: Large radius residual")
    
    return test_img, result


def validate_data_format():
    """Validate that data is in the correct format"""
    
    print("Validating data format...")
    
    # Create a dummy TestImage
    test_img = TestImage(
        image=np.zeros((512, 512), dtype=np.uint8),
        position_camera=np.array([0, 0, 150e6]),
        velocity_camera=np.array([0, 0, 0]),
        attitude_camera_to_inertial=np.eye(3),
        sun_direction_camera=np.array([1, 0, 0]),
        sun_direction_inertial=np.array([1, 0, 0]),
        body_radii=np.array([1737400, 1737400, 1737400]),
        body_orientation=np.eye(3),
        target_body=TargetBody.MOON,
        camera_K=np.array([[2000, 0, 256], [0, 2000, 256], [0, 0, 1]]),
        distortion={'k1': 0, 'k2': 0, 'k3': 0, 'p1': 0, 'p2': 0},
        timestamp=0.0,
        image_id="validation_001",
        pass_id="Validation",
        range_m=150e6,
        phase_angle_deg=90.0
    )
    
    print("✓ TestImage format is valid")
    
    # Create a dummy CRAResult
    result = CRAResult(
        position_camera=np.array([0, 0, 150e6]),
        success=True
    )
    
    print("✓ CRAResult format is valid")
    
    # Try adding to tester
    tester = setup_tester()
    tester.add_test_case(test_img, result)
    
    print("✓ Can add test cases to tester")
    print("\nData format validation passed!")


if __name__ == "__main__":
    # Option 1: Test single image first (recommended)
    # test_single_image()
    
    # Option 2: Validate data format
    # validate_data_format()
    
    # Option 3: Run full pipeline
    main()