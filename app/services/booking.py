import logging
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Booking
from app.schemas import BookingCreate

logger = logging.getLogger(__name__)

# Tool definition that we give to the LLM. The LLM decides when to call it,
# we only save the booking after validating the fields ourselves.
BOOK_INTERVIEW_TOOL_NAME = "book_interview"

BOOK_INTERVIEW_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": BOOK_INTERVIEW_TOOL_NAME,
        "description": (
            "Book an interview for the user. Only call this when the user has clearly asked to book "
            "an interview AND has given all four details: full name, email, date and time. "
            "Never guess or make up any of these values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Full name of the candidate"},
                "email": {"type": "string", "description": "Email address of the candidate"},
                "date": {"type": "string", "description": "Interview date in YYYY-MM-DD format"},
                "time": {"type": "string", "description": "Interview time in 24h HH:MM format"},
            },
            "required": ["name", "email", "date", "time"],
        },
    },
}


@dataclass
class BookingResult:
    success: bool
    message: str  # this message is sent back to the LLM as the tool result
    booking: Booking | None = None


def create_booking(db: Session, session_id: str, tool_input: dict[str, Any]) -> BookingResult:
    try:
        data = BookingCreate(**tool_input)
    except ValidationError as e:
        errors = "; ".join(f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors())
        logger.info("Booking validation failed: %s", errors)
        return BookingResult(success=False, message=f"Error: booking failed, invalid details: {errors}")

    # don't allow two interviews in the same slot
    existing = db.scalar(
        select(Booking).where(Booking.interview_date == data.date, Booking.interview_time == data.time)
    )
    if existing is not None:
        return BookingResult(
            success=False,
            message=f"Error: the slot {data.date} {data.time.strftime('%H:%M')} is already booked. Ask the user for another time.",
        )

    booking = Booking(
        session_id=session_id,
        name=data.name,
        email=str(data.email),
        interview_date=data.date,
        interview_time=data.time,
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)

    logger.info("Created booking %d for %s", booking.id, booking.email)
    return BookingResult(
        success=True,
        message=(
            f"Interview booked successfully (booking id {booking.id}) for {booking.name} "
            f"<{booking.email}> on {booking.interview_date} at {booking.interview_time.strftime('%H:%M')}."
        ),
        booking=booking,
    )


def list_bookings(db: Session) -> list[Booking]:
    return list(db.scalars(select(Booking).order_by(Booking.interview_date, Booking.interview_time)))
