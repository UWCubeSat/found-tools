#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "common/spatial/camera.hpp"
// camera.hpp transitively includes attitude-utils.hpp and decimal.hpp

namespace py = pybind11;

using namespace found;

PYBIND11_MODULE(_cameraGeometry, m) {
    m.doc() = "Python bindings for the found camera geometry and attitude utilities";

    // ---- Vec2 ----
    py::class_<Vec2>(m, "Vec2")
        .def(py::init([](decimal x, decimal y) { return Vec2{x, y}; }),
             py::arg("x") = decimal{0}, py::arg("y") = decimal{0})
        .def_readwrite("x", &Vec2::x)
        .def_readwrite("y", &Vec2::y)
        .def("Magnitude",   &Vec2::Magnitude)
        .def("MagnitudeSq", &Vec2::MagnitudeSq)
        .def("Normalize",   &Vec2::Normalize)
        .def("__mul__", [](const Vec2 &a, const Vec2 &b) { return a * b; })
        .def("__mul__", [](const Vec2 &a, decimal s)     { return a * s; })
        .def("__add__", [](const Vec2 &a, const Vec2 &b) { return a + b; })
        .def("__sub__", [](const Vec2 &a, const Vec2 &b) { return a - b; });

    // ---- Vec3 ----
    py::class_<Vec3>(m, "Vec3")
        .def(py::init<>())
        .def(py::init<decimal, decimal, decimal>(),
             py::arg("x"), py::arg("y"), py::arg("z"))
        .def_readwrite("x", &Vec3::x)
        .def_readwrite("y", &Vec3::y)
        .def_readwrite("z", &Vec3::z)
        .def("Magnitude",    &Vec3::Magnitude)
        .def("MagnitudeSq",  &Vec3::MagnitudeSq)
        .def("Normalize",    &Vec3::Normalize)
        .def("CrossProduct", &Vec3::CrossProduct)
        .def("OuterProduct", &Vec3::OuterProduct)
        .def("__neg__",     [](const Vec3 &a)                { return -a; })
        .def("__sub__",     [](const Vec3 &a, const Vec3 &b) { return a - b; })
        .def("__mul__",     [](const Vec3 &a, const Vec3 &b) { return a * b; })
        .def("__mul__",     [](const Vec3 &a, decimal s)     { return a * s; })
        .def("__truediv__", [](const Vec3 &a, decimal s)     { return a / s; })
        .def("__iadd__",    [](Vec3 &a, const Vec3 &b) -> Vec3 { a += b; return a; });

    // ---- Mat3 ----
    py::class_<Mat3>(m, "Mat3")
        .def(py::init<>())
        .def(py::init([](std::vector<decimal> const &v) {
                if (static_cast<int>(v.size()) != 9)
                    throw std::invalid_argument("Mat3 requires exactly 9 elements");
                Mat3 out{};
                for (int i = 0; i < 9; ++i) out.x[i] = v[i];
                return out;
            }), py::arg("elements"))
        .def("At",        &Mat3::At)
        .def("Column",    &Mat3::Column)
        .def("Row",       &Mat3::Row)
        .def("Trace",     &Mat3::Trace)
        .def("Det",       &Mat3::Det)
        .def("Transpose", &Mat3::Transpose)
        .def("Inverse",   &Mat3::Inverse)
        .def("__add__",    [](const Mat3 &a, const Mat3 &b) { return a + b; })
        .def("__mul__",    [](const Mat3 &a, const Mat3 &b) { return a * b; })
        .def("__matmul__", [](const Mat3 &a, const Vec3 &v) { return a * v; })
        .def("__mul__",    [](const Mat3 &a, decimal s)     { return a * s; });

    // ---- EulerAngles ----
    py::class_<EulerAngles>(m, "EulerAngles")
        .def(py::init<decimal, decimal, decimal>(),
             py::arg("ra")   = decimal{0},
             py::arg("de")   = decimal{0},
             py::arg("roll") = decimal{0})
        .def_readwrite("ra",   &EulerAngles::ra)
        .def_readwrite("de",   &EulerAngles::de)
        .def_readwrite("roll", &EulerAngles::roll);

    // ---- Quaternion ----
    py::class_<Quaternion>(m, "Quaternion")
        .def(py::init<>())
        .def(py::init<decimal, decimal, decimal, decimal>(),
             py::arg("real"), py::arg("i"), py::arg("j"), py::arg("k"))
        .def(py::init<const Vec3 &, decimal>(),
             py::arg("axis"), py::arg("theta"))
        .def(py::init<const Vec3 &>(),
             py::arg("vector"))
        .def_readwrite("real", &Quaternion::real)
        .def_readwrite("i",    &Quaternion::i)
        .def_readwrite("j",    &Quaternion::j)
        .def_readwrite("k",    &Quaternion::k)
        .def("Conjugate",    &Quaternion::Conjugate)
        .def("Vector",       &Quaternion::Vector)
        .def("SetVector",    &Quaternion::SetVector)
        .def("Rotate",       &Quaternion::Rotate)
        .def("Angle",        &Quaternion::Angle)
        .def("SetAngle",     &Quaternion::SetAngle)
        .def("IsUnit",       &Quaternion::IsUnit)
        .def("Canonicalize", &Quaternion::Canonicalize)
        .def("ToSpherical",  &Quaternion::ToSpherical)
        .def("__mul__", [](const Quaternion &a, const Quaternion &b) { return a * b; })
        .def("__neg__", [](const Quaternion &a)                      { return -a; });

    // ---- Attitude ----
    py::class_<Attitude>(m, "Attitude")
        .def(py::init<>())
        .def(py::init<const Quaternion &>(), py::arg("quat"))
        .def(py::init<const Mat3 &>(),       py::arg("matrix"))
        .def("GetQuaternion", &Attitude::GetQuaternion)
        .def("GetDCM",        &Attitude::GetDCM)
        .def("ToSpherical",   &Attitude::ToSpherical)
        .def("Rotate",        &Attitude::Rotate);

    // ---- Camera ----
    py::class_<Camera>(m, "Camera")
        .def(py::init<decimal, decimal, decimal, decimal, int, int>(),
             py::arg("focalLength"), py::arg("pixelSize"),
             py::arg("xCenter"),     py::arg("yCenter"),
             py::arg("xResolution"), py::arg("yResolution"))
        .def(py::init<decimal, decimal, int, int>(),
             py::arg("focalLength"), py::arg("pixelSize"),
             py::arg("xResolution"), py::arg("yResolution"))
        .def("SpatialToCamera", &Camera::SpatialToCamera)
        .def("CameraToSpatial", &Camera::CameraToSpatial)
        .def("InSensor",        &Camera::InSensor)
        .def("XResolution",     &Camera::XResolution)
        .def("YResolution",     &Camera::YResolution)
        .def("FocalLength",     &Camera::FocalLength)
        .def("PixelSize",       &Camera::PixelSize)
        .def("Fov",             &Camera::Fov)
        .def("SetFocalLength",  &Camera::SetFocalLength);

    // ---- Free functions ----
    m.def("SphericalToQuaternion",
          static_cast<Quaternion (*)(decimal, decimal, decimal)>(&SphericalToQuaternion),
          py::arg("ra"), py::arg("dec"), py::arg("roll"));
    m.def("SphericalToQuaternion",
          static_cast<Quaternion (*)(EulerAngles)>(&SphericalToQuaternion),
          py::arg("angles"));
    m.def("FovToFocalLength",
          &FovToFocalLength,
          py::arg("xFov"), py::arg("xResolution"));
    m.def("FocalLengthToFov",
          &FocalLengthToFov,
          py::arg("focalLength"), py::arg("xResolution"), py::arg("pixelSize"));
}
