import React from 'react';
import {
  Activity, ArrowDown, ArrowUp, ArrowUpDown, AtSign, Briefcase, Building, CalendarDays, Camera, Check, ChevronDown, ChevronRight, ChevronUp, Clock,
  IdCard, Smartphone,
  Columns3, Command, Contact, Copy, CopyCheck, CornerDownLeft, Database, Download, Ellipsis, Eraser, ExternalLink, Eye, FileText,
  Flame, GitMerge, ImagePlus, Info, KeyRound, Keyboard, LayoutDashboard, Link, List, ListPlus, LogOut, Mail, Megaphone, Menu,
  Moon, Paperclip, Pencil, ChevronsUpDown, Maximize2, Minimize2, Phone, Play, Plus, QrCode, RefreshCw, Rows3, Save, Search, Send, Server, Shield, SlidersHorizontal,
  Sparkles, StickyNote, Store, Sun, Trash, TriangleAlert, Undo2, Upload, User, UserPlus, Users, Webhook, Workflow, X, Zap,
  Map as MapIcon, DoorOpen,
  Globe, MapPin, Ticket, Shirt, Tag, Flag,
  Mic, Square, Image as ImageIcon, Loader2, MessagesSquare, PanelLeftClose, PanelLeftOpen, AudioLines,
  Package, Lightbulb, Trophy, MessageSquareQuote, Target, ShieldQuestionMark,
  type LucideIcon,
} from 'lucide-react';
import { ICON } from './tokens';

// Every static icon is Lucide (stroke-based, MIT, same names as lucide-react-native in the app).
// Size and stroke come from ui/tokens — never set per icon.
const ICONS = {
  dashboard: LayoutDashboard, activity: Activity, users: Users, contacts: Contact, duplicate: CopyCheck, list: List,
  pipeline: Workflow, megaphone: Megaphone, store: Store, file: FileText, sparkles: Sparkles, server: Server,
  download: Download, upload: Upload, plus: Plus, search: Search, x: X, check: Check, chevronRight: ChevronRight,
  chevronDown: ChevronDown, trash: Trash, pencil: Pencil, refresh: RefreshCw, sun: Sun, moon: Moon, logout: LogOut,
  alert: TriangleAlert, info: Info, flame: Flame, mail: Mail, phone: Phone, link: Link, qr: QrCode, play: Play,
  merge: GitMerge, shield: Shield, database: Database, webhook: Webhook, send: Send, calendar: CalendarDays, bolt: Zap,
  eye: Eye, external: ExternalLink, key: KeyRound, user: User, menu: Menu, clock: Clock, copy: Copy, more: Ellipsis,
  command: Command, keyboard: Keyboard, enter: CornerDownLeft, arrowUp: ArrowUp, arrowDown: ArrowDown,
  columns: Columns3, rows: Rows3, sort: ArrowUpDown, paperclip: Paperclip, imagePlus: ImagePlus, userPlus: UserPlus,
  sliders: SlidersHorizontal, save: Save, undo: Undo2, briefcase: Briefcase, note: StickyNote, chevronUp: ChevronUp,
  listPlus: ListPlus, eraser: Eraser, camera: Camera, building: Building, atSign: AtSign, smartphone: Smartphone, idCard: IdCard,
  chevronsUpDown: ChevronsUpDown, maximize: Maximize2, minimize: Minimize2,
  globe: Globe, mapPin: MapPin, ticket: Ticket, shirt: Shirt, tag: Tag, flag: Flag,
  map: MapIcon, door: DoorOpen,
  mic: Mic, stop: Square, image: ImageIcon, loader: Loader2, chats: MessagesSquare, panelClose: PanelLeftClose, panelOpen: PanelLeftOpen, audio: AudioLines,
  package: Package, lightbulb: Lightbulb, trophy: Trophy, quote: MessageSquareQuote, target: Target, objection: ShieldQuestionMark,
} satisfies Record<string, LucideIcon>;

export type IconName = keyof typeof ICONS;

export function Icon({ name, size = ICON.size, className = '', strokeWidth = ICON.stroke }: { name: IconName; size?: number; className?: string; strokeWidth?: number }) {
  const C = ICONS[name];
  return <C size={size} strokeWidth={strokeWidth} className={`shrink-0 ${className}`} aria-hidden="true" />;
}
