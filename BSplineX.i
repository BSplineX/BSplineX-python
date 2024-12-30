%feature("autodoc", "2");
%typemap(doc) size_t "int"
%typemap(doc) double "float"
%typemap(doc) std::vector<double> "list[float]"
%typemap(doc) const std::vector<double>& "list[float]"
// %feature("python:annotations", "c");
%include BSplineX/bindings/BSplineX.i