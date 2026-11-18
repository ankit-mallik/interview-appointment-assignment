import datetime as dt

from sqlalchemy.orm import Session

from app.services.booking import create_booking, list_bookings


def tomorrow() -> str:
    return (dt.date.today() + dt.timedelta(days=1)).isoformat()


def test_create_booking_success(db: Session) -> None:
    result = create_booking(
        db, "session-1", {"name": "John Doe", "email": "john@example.com", "date": tomorrow(), "time": "10:30"}
    )

    assert result.success
    assert result.booking is not None
    assert result.booking.email == "john@example.com"
    assert len(list_bookings(db)) == 1


def test_create_booking_invalid_email(db: Session) -> None:
    result = create_booking(
        db, "session-1", {"name": "John Doe", "email": "not-an-email", "date": tomorrow(), "time": "10:30"}
    )

    assert not result.success
    assert "email" in result.message
    assert list_bookings(db) == []


def test_create_booking_date_in_past(db: Session) -> None:
    result = create_booking(
        db, "session-1", {"name": "John Doe", "email": "john@example.com", "date": "2020-01-01", "time": "10:30"}
    )

    assert not result.success
    assert "past" in result.message


def test_create_booking_same_slot_twice(db: Session) -> None:
    details = {"name": "John Doe", "email": "john@example.com", "date": tomorrow(), "time": "10:30"}
    assert create_booking(db, "session-1", details).success

    second = create_booking(db, "session-2", {**details, "name": "Jane Doe", "email": "jane@example.com"})
    assert not second.success
    assert "already booked" in second.message
