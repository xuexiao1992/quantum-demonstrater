# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 18:36:42 2018

@author: TUD278306
"""

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, InlineQueryHandler
from telegram import InlineQueryResultArticle, InputTextMessageContent
 
import logging
from datetime import datetime
import time
 
from functools import wraps
from PIL import ImageGrab
import os
 
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)
 
token = '739604073:AAGiaEu69l5mNZOwjsM_n11yMDoh8z-Lhes'
 
ALLOWEDCHATS = [94078631] # enter which chats are allowed to interact with the bot
 
updater = Updater(token)
dispatcher = updater.dispatcher

def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="Welcome to the F006 bot!")

def gate_B(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="B: {} mV".format(G.B()))

def gate_LD(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="LD: {} mV".format(G.LD()))

def gate_LP(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="LP: {} mV".format(LP()))

def gate_LS(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="LS: {} mV".format(G.LS()))

def gate_T(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="T: {} mV".format(T()))

def gate_SQD1(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="SQD1: {} mV".format(G.SQD1()))

def gate_SQD3(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="SQD3: {} mV".format(G.SQD3()))

def gate_values(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="Gate values:".format(G.allvalues()))

def plot(bot, update):
    plot1D(experiment.data_set)

def unknown(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="Sorry, I didn't understand that command.")
   
updater.start_polling()
 
start_handler = CommandHandler('start', start)
B_handler = CommandHandler('gate_B', gate_B)
LD_handler = CommandHandler('gate_LD', gate_LD)
LP_handler = CommandHandler('gate_LP', gate_LP)
LS_handler = CommandHandler('gate_LS', gate_LS)
T_handler = CommandHandler('gate_T', gate_T)
SQD1_handler = CommandHandler('gate_SQD1', gate_SQD1)
SQD3_handler = CommandHandler('gate_SQD3', gate_SQD3)
gates_handler = CommandHandler('gate_values', gate_values)
plot_handler = CommandHandler('plot',plot)
unknown_handler = MessageHandler(Filters.command, unknown)
 
dispatcher.add_handler(start_handler)
dispatcher.add_handler(B_handler)
dispatcher.add_handler(LD_handler)
dispatcher.add_handler(LP_handler)
dispatcher.add_handler(LS_handler)
dispatcher.add_handler(T_handler)
dispatcher.add_handler(SQD1_handler)
dispatcher.add_handler(SQD3_handler)
dispatcher.add_handler(gates_handler)
dispatcher.add_handler(plot_handler)
dispatcher.add_handler(unknown_handler)
 
#updater.stop()