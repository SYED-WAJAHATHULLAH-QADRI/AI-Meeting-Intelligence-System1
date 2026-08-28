# ============================================================
# AI MEETING INTELLIGENCE SYSTEM
# Streamlit Cloud / Faster-Whisper version
# ============================================================

import json
import os
import tempfile
import time
from pathlib import Path
from typing import List, Literal, Optional

import streamlit as st

from faster_whisper import WhisperModel
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="AI Meeting Intelligence",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# OUTPUT SCHEMA
# ============================================================

class Decision(BaseModel):
    decision: str = Field(
        description="A confirmed meeting decision."
    )

    evidence: str = Field(
        description="Exact quotation supporting the decision."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


class ActionItem(BaseModel):
    action: str = Field(
        description="A meeting task or follow-up action."
    )

    owner: Optional[str] = Field(
        default=None,
        description="Responsible person or null."
    )

    deadline: Optional[str] = Field(
        default=None,
        description="Stated deadline or null."
    )

    status: Literal[
        "assigned",
        "proposed",
        "unclear"
    ]

    evidence: str = Field(
        description="Exact quotation supporting the action."
    )

    confidence: float = Field(
        ge=0.0,
        le=1.0
    )


class MeetingIntelligence(BaseModel):
    meeting_title: str
    summary: str

    key_topics: List[str] = Field(
        default_factory=list
    )

    decisions: List[Decision] = Field(
        default_factory=list
    )

    action_items: List[ActionItem] = Field(
        default_factory=list
    )

    unresolved_issues: List[str] = Field(
        default_factory=list
    )

    ambiguities: List[str] = Field(
        default_factory=list
    )


# ============================================================
# PROMPTS
# ============================================================

GENERIC_PROMPT = """
Analyse the following meeting transcript.

Identify:
- the meeting title;
- summary;
- key topics;
- decisions;
- action items;
- owners;
- deadlines;
- unresolved issues; and
- ambiguities.

Use only information contained in the transcript.

TRANSCRIPT:
{transcript}
"""


STRUCTURED_PROMPT = """
You are a highly cautious, context-aware meeting-intelligence
extraction system.

Analyse only the supplied transcript.

DEFINITIONS

A confirmed decision is an outcome clearly agreed, approved,
accepted, selected or confirmed by the meeting participants.

A suggestion, question, personal opinion, possibility, rejected
proposal or unresolved matter is not a confirmed decision.

An assigned action is a task explicitly assigned to or accepted
by a person or team.

A proposed action is a task suggested but not clearly assigned
or accepted.

An unclear action is a possible task where responsibility or
commitment cannot safely be determined.

RULES

1. Read the complete transcript before extracting information.
2. Consider context across multiple speaker turns.
3. Separate confirmed decisions from suggestions and discussions.
4. Do not convert general discussion into a decision.
5. Never invent owners, deadlines, dates, tasks or decisions.
6. Use null when an owner or deadline is not explicitly stated.
7. Preserve dates and deadlines exactly as expressed.
8. Evidence must be copied exactly from the transcript.
9. Evidence must directly support the extracted item.
10. Put unanswered matters under unresolved_issues.
11. Put uncertain interpretations under ambiguities.
12. Do not use outside knowledge.
13. Prefer omission over an unsupported prediction.
14. Use empty lists when no supported items exist.

Before returning the result, verify:

- Every decision is genuinely confirmed.
- Every action is assigned, proposed or unclear.
- Every owner belongs to the correct task.
- Every deadline belongs to the correct task.
- Every evidence quotation appears in the transcript.
- No information has been invented.

TRANSCRIPT:
{transcript}
"""


# ============================================================
# TEXT PROCESSING
# ============================================================

def clean_transcript(text):
    if not isinstance(text, str):
        raise TypeError(
            "Transcript must be text."
        )

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = text.replace("\u00a0", " ")

    cleaned_lines = []

    for line in text.splitlines():
        line = " ".join(line.split()).strip()

        if line:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()

    if not cleaned:
        raise ValueError(
            "The transcript is empty."
        )

    return cleaned


def normalise_for_evidence(text):
    return " ".join(
        str(text).lower().split()
    )


# ============================================================
# FASTER-WHISPER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_whisper_model():
    """
    Load an English Whisper model optimised for CPU deployment.
    """

    return WhisperModel(
        "base.en",
        device="cpu",
        compute_type="int8",
        cpu_threads=2,
        num_workers=1
    )


def transcribe_uploaded_audio(uploaded_file):
    suffix = Path(
        uploaded_file.name
    ).suffix.lower()

    supported_suffixes = {
        ".wav",
        ".mp3",
        ".m4a",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".webm"
    }

    if suffix not in supported_suffixes:
        suffix = ".wav"

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            suffix=suffix
        ) as temporary_file:

            temporary_file.write(
                uploaded_file.getbuffer()
            )

            temporary_file.flush()

            temporary_path = temporary_file.name

        temporary_file_path = Path(
            temporary_path
        )

        if not temporary_file_path.exists():
            raise FileNotFoundError(
                "Temporary audio file was not created."
            )

        if temporary_file_path.stat().st_size == 0:
            raise ValueError(
                "The uploaded audio file is empty."
            )

        model = load_whisper_model()

        started = time.perf_counter()

        segments, information = model.transcribe(
            temporary_path,
            language="en",
            beam_size=5,
            best_of=5,
            patience=1.0,
            temperature=0.0,
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 500
            },
            initial_prompt=(
                "This is a professional meeting containing "
                "decisions, tasks, responsible owners, deadlines, "
                "dates, project names and follow-up actions."
            )
        )

        segment_records = []
        transcript_parts = []

        for segment in segments:
            segment_text = segment.text.strip()

            if not segment_text:
                continue

            transcript_parts.append(
                segment_text
            )

            segment_records.append({
                "start": round(
                    segment.start,
                    2
                ),
                "end": round(
                    segment.end,
                    2
                ),
                "text": segment_text
            })

        processing_seconds = (
            time.perf_counter() - started
        )

        transcript = clean_transcript(
            " ".join(transcript_parts)
        )

        return {
            "transcript": transcript,
            "segments": segment_records,
            "processing_seconds": round(
                processing_seconds,
                3
            ),
            "model": "base.en",
            "implementation": "Faster-Whisper",
            "device": "CPU",
            "compute_type": "int8",
            "language": information.language,
            "language_probability": round(
                information.language_probability,
                3
            )
        }

    finally:
        if temporary_path:
            temporary_file_path = Path(
                temporary_path
            )

            if temporary_file_path.exists():
                temporary_file_path.unlink()


# ============================================================
# EVIDENCE VERIFICATION
# ============================================================

def verify_evidence(result, transcript):
    """
    Remove extracted decisions or actions when their evidence
    cannot be located in the transcript.
    """

    normalised_transcript = (
        normalise_for_evidence(
            transcript
        )
    )

    verified_decisions = []
    verified_actions = []
    rejected_items = []

    for decision in result.get(
        "decisions",
        []
    ):
        evidence = normalise_for_evidence(
            decision.get(
                "evidence",
                ""
            )
        )

        if (
            evidence
            and evidence in normalised_transcript
        ):
            decision[
                "evidence_verified"
            ] = True

            verified_decisions.append(
                decision
            )

        else:
            rejected_items.append(
                "Unsupported decision removed during "
                "evidence validation: "
                + decision.get(
                    "decision",
                    ""
                )
            )

    for action in result.get(
        "action_items",
        []
    ):
        evidence = normalise_for_evidence(
            action.get(
                "evidence",
                ""
            )
        )

        if (
            evidence
            and evidence in normalised_transcript
        ):
            action[
                "evidence_verified"
            ] = True

            verified_actions.append(
                action
            )

        else:
            rejected_items.append(
                "Unsupported action removed during "
                "evidence validation: "
                + action.get(
                    "action",
                    ""
                )
            )

    result["decisions"] = (
        verified_decisions
    )

    result["action_items"] = (
        verified_actions
    )

    result.setdefault(
        "ambiguities",
        []
    ).extend(rejected_items)

    return result


# ============================================================
# GEMINI EXTRACTION
# ============================================================

def extract_meeting(
    transcript,
    api_key,
    model_name,
    prompt_type
):
    transcript = clean_transcript(
        transcript
    )

    prompt_template = (
        STRUCTURED_PROMPT
        if prompt_type == "Structured"
        else GENERIC_PROMPT
    )

    prompt = prompt_template.format(
        transcript=transcript
    )

    client = genai.Client(
        api_key=api_key
    )

    started = time.perf_counter()

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type=(
                "application/json"
            ),
            response_schema=(
                MeetingIntelligence
            )
        )
    )

    processing_seconds = (
        time.perf_counter() - started
    )

    if response.parsed is not None:
        if isinstance(
            response.parsed,
            MeetingIntelligence
        ):
            validated_result = (
                response.parsed
            )

        else:
            validated_result = (
                MeetingIntelligence.model_validate(
                    response.parsed
                )
            )

    else:
        validated_result = (
            MeetingIntelligence.model_validate_json(
                response.text
            )
        )

    verified_result = verify_evidence(
        validated_result.model_dump(),
        transcript
    )

    return {
        "result": verified_result,
        "processing_seconds": round(
            processing_seconds,
            3
        ),
        "schema_valid": True,
        "evidence_validation": True,
        "model": model_name,
        "temperature": 0.2,
        "prompt_type": prompt_type
    }


# ============================================================
# REPORT EXPORT
# ============================================================

def create_markdown_report(
    result_record,
    transcript
):
    result = result_record["result"]

    lines = [
        f"# {result['meeting_title']}",
        "",
        "## System Information",
        "",
        f"- Prompt: {result_record['prompt_type']}",
        f"- Gemini model: {result_record['model']}",
        f"- Temperature: {result_record['temperature']}",
        "- Schema validation: Passed",
        "- Evidence validation: Applied",
        "",
        "## Summary",
        "",
        result["summary"],
        "",
        "## Key Topics",
        ""
    ]

    if result["key_topics"]:
        for topic in result["key_topics"]:
            lines.append(
                f"- {topic}"
            )
    else:
        lines.append(
            "- None identified"
        )

    lines.extend([
        "",
        "## Decisions",
        ""
    ])

    if result["decisions"]:
        for number, decision in enumerate(
            result["decisions"],
            start=1
        ):
            lines.extend([
                f"### Decision {number}",
                "",
                decision["decision"],
                "",
                f"**Evidence:** "
                f"{decision['evidence']}",
                "",
                f"**Confidence:** "
                f"{decision['confidence']:.2f}",
                ""
            ])
    else:
        lines.append(
            "No confirmed decisions identified."
        )

    lines.extend([
        "",
        "## Action Items",
        ""
    ])

    if result["action_items"]:
        for number, action in enumerate(
            result["action_items"],
            start=1
        ):
            lines.extend([
                f"### Action {number}",
                "",
                f"**Task:** {action['action']}",
                "",
                f"**Owner:** "
                f"{action.get('owner') or 'Not stated'}",
                "",
                f"**Deadline:** "
                f"{action.get('deadline') or 'Not stated'}",
                "",
                f"**Status:** {action['status']}",
                "",
                f"**Evidence:** "
                f"{action['evidence']}",
                ""
            ])
    else:
        lines.append(
            "No action items identified."
        )

    lines.extend([
        "",
        "## Unresolved Issues",
        ""
    ])

    if result["unresolved_issues"]:
        for issue in result[
            "unresolved_issues"
        ]:
            lines.append(f"- {issue}")
    else:
        lines.append("- None identified")

    lines.extend([
        "",
        "## Ambiguities",
        ""
    ])

    if result["ambiguities"]:
        for ambiguity in result[
            "ambiguities"
        ]:
            lines.append(
                f"- {ambiguity}"
            )
    else:
        lines.append("- None identified")

    lines.extend([
        "",
        "## Reviewed Transcript",
        "",
        transcript,
        "",
        "---",
        "",
        "AI-generated draft. Human review is required."
    ])

    return "\n".join(lines)


# ============================================================
# SESSION STATE
# ============================================================

session_defaults = {
    "transcript": "",
    "whisper_record": None,
    "result_record": None
}

for key, default_value in (
    session_defaults.items()
):
    if key not in st.session_state:
        st.session_state[key] = (
            default_value
        )


# ============================================================
# SECURE API KEY
# ============================================================

try:
    GEMINI_API_KEY = st.secrets[
        "GEMINI_API_KEY"
    ]

except Exception:
    GEMINI_API_KEY = os.getenv(
        "GEMINI_API_KEY"
    )


# ============================================================
# INTERFACE
# ============================================================

st.title(
    "🧠 AI Meeting Intelligence System"
)

st.write(
    "Convert meeting audio or transcript text into "
    "structured, reviewable decisions and actions."
)

st.warning(
    "AI-generated results are drafts. Check all decisions, "
    "owners and deadlines against the original meeting."
)


with st.sidebar:
    st.header("System Settings")

    input_type = st.radio(
        "Meeting input",
        [
            "Audio file",
            "Transcript text"
        ]
    )

    prompt_type = st.selectbox(
        "Prompt method",
        [
            "Structured",
            "Generic"
        ]
    )

    model_name = st.selectbox(
        "Gemini model",
        [
            "gemini-2.5-flash",
            "gemini-3-flash-preview",
            "gemini-2.0-flash-001"
        ],
        index=0
    )

    st.caption(
        "Temperature: 0.2"
    )

    st.divider()
    st.subheader("System status")

    st.success(
        "Faster-Whisper configured"
    )

    if GEMINI_API_KEY:
        st.success(
            "Gemini key configured"
        )
    else:
        st.error(
            "Gemini key missing"
        )


# ============================================================
# INPUT
# ============================================================

if input_type == "Audio file":
    uploaded_audio = st.file_uploader(
        "Upload meeting audio",
        type=[
            "wav",
            "mp3",
            "m4a",
            "mp4",
            "mpeg",
            "mpga",
            "webm"
        ]
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if st.button(
            "Transcribe audio",
            type="primary",
            use_container_width=True
        ):
            try:
                with st.spinner(
                    "Downloading/loading Whisper model and "
                    "transcribing audio..."
                ):
                    whisper_record = (
                        transcribe_uploaded_audio(
                            uploaded_audio
                        )
                    )

                st.session_state[
                    "whisper_record"
                ] = whisper_record

                st.session_state[
                    "transcript"
                ] = whisper_record[
                    "transcript"
                ]

                st.session_state[
                    "result_record"
                ] = None

                st.success(
                    "Whisper transcription completed."
                )

            except Exception as error:
                st.error(
                    f"Transcription failed: {error}"
                )

else:
    pasted_transcript = st.text_area(
        "Paste meeting transcript",
        value=st.session_state[
            "transcript"
        ],
        height=280,
        placeholder=(
            "Aisha: We agreed to retain the launch date.\n"
            "Manager: Bilal, complete the test by Friday."
        )
    )

    if pasted_transcript.strip():
        try:
            st.session_state[
                "transcript"
            ] = clean_transcript(
                pasted_transcript
            )

        except ValueError:
            pass


# ============================================================
# TRANSCRIPT REVIEW
# ============================================================

if st.session_state["transcript"]:
    st.divider()
    st.subheader("Transcript review")

    reviewed_transcript = st.text_area(
        "Review and correct the transcript",
        value=st.session_state[
            "transcript"
        ],
        height=300,
        key="reviewed_transcript"
    )

    if st.session_state[
        "whisper_record"
    ]:
        whisper_record = st.session_state[
            "whisper_record"
        ]

        first, second, third = (
            st.columns(3)
        )

        first.metric(
            "Whisper model",
            whisper_record["model"]
        )

        second.metric(
            "Transcription time",
            f"{whisper_record['processing_seconds']:.2f}s"
        )

        third.metric(
            "Language confidence",
            whisper_record[
                "language_probability"
            ]
        )

    if st.button(
        "Generate meeting intelligence",
        type="primary",
        use_container_width=True
    ):
        if not GEMINI_API_KEY:
            st.error(
                "Add GEMINI_API_KEY through "
                "Manage app → Settings → Secrets."
            )

        else:
            try:
                cleaned_transcript = (
                    clean_transcript(
                        reviewed_transcript
                    )
                )

                with st.spinner(
                    "Running Gemini extraction..."
                ):
                    result_record = (
                        extract_meeting(
                            transcript=(
                                cleaned_transcript
                            ),
                            api_key=(
                                GEMINI_API_KEY
                            ),
                            model_name=(
                                model_name
                            ),
                            prompt_type=(
                                prompt_type
                            )
                        )
                    )

                st.session_state[
                    "transcript"
                ] = cleaned_transcript

                st.session_state[
                    "result_record"
                ] = result_record

                st.success(
                    "Extraction and validation completed."
                )

            except Exception as error:
                st.error(
                    f"Gemini extraction failed: {error}"
                )


# ============================================================
# RESULTS
# ============================================================

if st.session_state["result_record"]:
    result_record = st.session_state[
        "result_record"
    ]

    result = result_record["result"]

    st.divider()
    st.header(
        result["meeting_title"]
    )

    first, second, third = st.columns(3)

    first.metric(
        "Gemini time",
        f"{result_record['processing_seconds']:.2f}s"
    )

    second.metric(
        "Prompt",
        result_record["prompt_type"]
    )

    third.metric(
        "Validation",
        "Passed"
    )

    (
        summary_tab,
        decisions_tab,
        actions_tab,
        review_tab,
        json_tab
    ) = st.tabs([
        "Summary",
        "Decisions",
        "Action Items",
        "Review",
        "JSON"
    ])

    with summary_tab:
        st.subheader("Summary")
        st.write(result["summary"])

        st.subheader("Key topics")

        if result["key_topics"]:
            for topic in result[
                "key_topics"
            ]:
                st.write(f"• {topic}")
        else:
            st.info(
                "No key topics identified."
            )

    with decisions_tab:
        if not result["decisions"]:
            st.info(
                "No confirmed decisions identified."
            )

        for number, decision in enumerate(
            result["decisions"],
            start=1
        ):
            with st.container(
                border=True
            ):
                st.markdown(
                    f"### Decision {number}"
                )

                st.write(
                    decision["decision"]
                )

                st.caption(
                    "Evidence: "
                    + decision["evidence"]
                )

                st.progress(
                    float(
                        decision["confidence"]
                    )
                )

    with actions_tab:
        if not result["action_items"]:
            st.info(
                "No action items identified."
            )

        for number, action in enumerate(
            result["action_items"],
            start=1
        ):
            with st.container(
                border=True
            ):
                st.markdown(
                    f"### Action {number}"
                )

                st.write(
                    action["action"]
                )

                owner_column, deadline_column = (
                    st.columns(2)
                )

                owner_column.write(
                    "**Owner:** "
                    + (
                        action.get("owner")
                        or "Not stated"
                    )
                )

                deadline_column.write(
                    "**Deadline:** "
                    + (
                        action.get(
                            "deadline"
                        )
                        or "Not stated"
                    )
                )

                st.write(
                    f"**Status:** "
                    f"{action['status']}"
                )

                st.caption(
                    "Evidence: "
                    + action["evidence"]
                )

                st.progress(
                    float(
                        action["confidence"]
                    )
                )

    with review_tab:
        st.subheader(
            "Unresolved issues"
        )

        if result["unresolved_issues"]:
            for issue in result[
                "unresolved_issues"
            ]:
                st.write(f"• {issue}")
        else:
            st.write("None identified.")

        st.subheader("Ambiguities")

        if result["ambiguities"]:
            for ambiguity in result[
                "ambiguities"
            ]:
                st.write(
                    f"• {ambiguity}"
                )
        else:
            st.write("None identified.")

        st.subheader(
            "Reviewed transcript"
        )

        st.text_area(
            "Source transcript",
            value=st.session_state[
                "transcript"
            ],
            height=250,
            disabled=True
        )

    with json_tab:
        st.json(result_record)


    # ========================================================
    # DOWNLOADS
    # ========================================================

    markdown_report = (
        create_markdown_report(
            result_record,
            st.session_state[
                "transcript"
            ]
        )
    )

    json_report = json.dumps(
        result_record,
        indent=2,
        ensure_ascii=False
    )

    st.subheader("Download reports")

    json_column, markdown_column = (
        st.columns(2)
    )

    json_column.download_button(
        "Download JSON report",
        data=json_report,
        file_name="meeting_report.json",
        mime="application/json",
        use_container_width=True
    )

    markdown_column.download_button(
        "Download Markdown report",
        data=markdown_report,
        file_name="meeting_report.md",
        mime="text/markdown",
        use_container_width=True
    )
