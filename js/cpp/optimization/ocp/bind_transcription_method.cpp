// Copyright (c) Sleipnir contributors

#include <emscripten/bind.h>
#include <sleipnir/optimization/ocp/transcription_method.hpp>

namespace em = emscripten;

namespace slp {

void bind_transcription_method(em::enum_<TranscriptionMethod>& e) {
  e.value("DIRECT_TRANSCRIPTION", TranscriptionMethod::DIRECT_TRANSCRIPTION);
  e.value("DIRECT_COLLOCATION", TranscriptionMethod::DIRECT_COLLOCATION);
  e.value("SINGLE_SHOOTING", TranscriptionMethod::SINGLE_SHOOTING);
}

}  // namespace slp
