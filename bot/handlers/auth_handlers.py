"""
Обработчики команд авторизации /admin и /moderator
"""

import logging
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message, ReplyKeyboardRemove
from aiogram.fsm.context import FSMContext

from bot.keyboards import get_admin_keyboard, get_main_keyboard, get_moderator_keyboard
from utils.auth_states import AuthStates
from utils.auth_system import (
    start_password_prompt,
    check_password,
    cancel_password_prompt,
    get_user_role,
    logout,
    get_session_info
)

logger = logging.getLogger(__name__)
router = Router(name="auth")


# ===== /admin =====
@router.message(Command("admin"))
async def cmd_admin_login(message: Message, state: FSMContext):
    user_id = message.from_user.id

    if get_user_role(user_id) == "admin":
        await message.answer(
            "✅ Вы уже авторизованы как администратор",
            reply_markup=get_admin_keyboard()
        )
        await show_admin_panel(message)
        return

    start_password_prompt(user_id, "admin")
    await state.set_state(AuthStates.waiting_for_password)

    await message.answer(
        "🔐 **Вход в админ-панель**\n\n"
        "Введите пароль администратора:\n\n"
        "_/cancel — отмена_",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardRemove()
    )


# ===== /moderator =====
@router.message(Command("moderator"))
async def cmd_moderator_login(message: Message, state: FSMContext):
    user_id = message.from_user.id

    if get_user_role(user_id) in ["admin", "moderator"]:
        await message.answer("✅ Вы уже авторизованы")
        return

    start_password_prompt(user_id, "moderator")
    await state.set_state(AuthStates.waiting_for_password)

    await message.answer(
        "🔐 **Вход в панель модератора**\n\n"
        "Введите пароль:\n\n"
        "_/cancel — отмена_",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardRemove()
    )


# ===== ВВОД ПАРОЛЯ (FSM) =====
@router.message(AuthStates.waiting_for_password)
async def handle_password(message: Message, state: FSMContext):
    user_id = message.from_user.id
    password = message.text.strip()

    # Пытаемся удалить сообщение с паролем
    try:
        await message.delete()
    except Exception:
        pass

    granted_role = check_password(user_id, password)

    if not granted_role:
        await message.answer(
            "❌ **Неверный пароль**\n\n"
            "Попробуйте ещё раз или /cancel",
            parse_mode="Markdown"
        )
        return

    await state.clear()

    if granted_role == "admin":
        await message.answer(
            "✅ **Авторизация успешна**",
            parse_mode="Markdown",
            reply_markup=get_admin_keyboard()
        )
        await show_admin_panel(message)

    elif granted_role == "moderator":
        await message.answer(
            "✅ **Авторизация успешна**",
            parse_mode="Markdown",
            reply_markup=get_moderator_keyboard() 
        )
        await show_moderator_panel(message)


# ===== /cancel =====
@router.message(Command("cancel"))
async def cmd_cancel(message: Message, state: FSMContext):
    if await state.get_state() is None:
        await message.answer("❓ Нечего отменять")
        return

    await state.clear()
    cancel_password_prompt(message.from_user.id)

    await message.answer(
        "❌ Ввод пароля отменён",
        reply_markup=get_main_keyboard()
    )


# ===== /logout =====
@router.message(Command("logout"))
async def cmd_logout(message: Message):
    user_id = message.from_user.id
    role = get_user_role(user_id)

    if role == "user":
        await message.answer("❌ Вы не авторизованы")
        return

    logout(user_id)

    import html
    await message.answer(
        f"👋 Вы вышли из роли <b>{html.escape(str(role))}</b>",
        parse_mode="HTML",
        reply_markup=get_main_keyboard()
    )


# ===== /whoami =====
@router.message(Command("whoami"))
async def cmd_whoami(message: Message):
    user_id = message.from_user.id
    role = get_user_role(user_id)
    session = get_session_info(user_id)

    if role == "user":
        await message.answer(
            "👤 **Роль:** пользователь\n\n"
            "/admin — вход админа\n"
            "/moderator — вход модератора",
            parse_mode="Markdown"
        )
        return

    import html

    text = f"👤 <b>Роль:</b> {html.escape(str(role).upper())}\n"
    if session:
        auth_at = html.escape(session['authorized_at'].strftime('%d.%m.%Y %H:%M'))
        time_left = html.escape(str(session.get('time_left_minutes', 'N/A')))
        text += (
            f"\n🕐 Авторизован: {auth_at}"
            f"\n⏳ Осталось: {time_left} мин"
        )

    await message.answer(text, parse_mode="HTML")


# ===== Панели =====
async def show_admin_panel(message: Message):
    await message.answer(
        "🔐 <b>Админ-панель</b>\n\n"
        "<b>Доступные команды:</b>\n\n"
        "📊 /stats_full - Полная статистика\n"
        "📈 /analytics - Аналитика\n"
        "⭐ /ratings - Рейтинги\n"
        "🚪 /logout - Выход"
    )


async def show_moderator_panel(message: Message):
    await message.answer(
        "🛡️ <b>Панель модератора</b>\n\n"
        "<b>Доступные команды:</b>\n\n"
        "📊 /mod_stats - Статистика за 7 дней\n"
        "⭐ /ratings - Рейтинги пользователей\n"
        "❓ /mod_popular - Топ-10 популярных вопросов\n"
        "👎 /mod_low_rated - Низкие оценки\n"
        "📥 /mod_export - Экспорт данных в CSV\n"
        "🚪 /logout - Выход из панели\n\n"
        "<b>Используйте кнопки ниже ⬇️</b>"
    )
