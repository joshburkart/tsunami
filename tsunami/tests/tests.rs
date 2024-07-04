use tsunami;

#[test]
fn test_sphere_no_crash() {
    let mut sphere = tsunami::geom::SphereGeometry::new(7, 6000);
    let _: Vec<_> = sphere.make_renderables(1).collect();
    for _ in 0..20 {
        sphere.step();
    }
}
