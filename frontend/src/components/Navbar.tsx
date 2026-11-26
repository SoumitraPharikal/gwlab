"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Menu, X, Sun, Moon } from "lucide-react";
import { useTheme } from "@/context/ThemeContext";
import { fetchSystemStatus } from "@/lib/api";

interface NavItem {
  name: string;
  href: string;
}

const NAV_ITEMS: NavItem[] = [
  { name: "Overview", href: "/" },
  { name: "Waveforms", href: "/waveforms" },
  { name: "Noise and PSD", href: "/noise-and-psd" },
  { name: "Matched filtering", href: "/matched-filtering" },
  { name: "Inference", href: "/inference" },
  { name: "Experiments", href: "/experiments" },
  { name: "Methods and provenance", href: "/methods" },
];

export function Navbar() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const { theme, toggleTheme } = useTheme();
  const [apiOnline, setApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let active = true;
    fetchSystemStatus()
      .then(() => {
        if (active) setApiOnline(true);
      })
      .catch(() => {
        if (active) setApiOnline(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/" || pathname === "/overview";
    return pathname.startsWith(href);
  };

  return (
    <nav className="sticky top-0 z-40 w-full border-b border-[var(--border)] bg-[var(--surface)] text-[var(--foreground)] transition-colors">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8 h-14">
        {/* Brand / Logo */}
        <div className="flex items-center gap-6">
          <Link
            href="/"
            className="flex items-center gap-2.5 text-[var(--foreground)] hover:opacity-80 transition-opacity"
          >
            {/* Minimal binary chirp waveform glyph */}
            <svg
              width="22"
              height="22"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.35"
              className="text-[var(--accent)] shrink-0"
              aria-hidden="true"
            >
              <path
                d="M 2 12 Q 5 7, 8 12 T 14 12 T 18 12 T 20.5 12 T 22 12"
                strokeLinecap="round"
              />
              <circle cx="21.5" cy="12" r="1.5" fill="currentColor" />
            </svg>
            <span className="font-serif text-[18px] tracking-tight font-normal text-[var(--foreground)]">
              GWLab
            </span>
          </Link>

          {/* Desktop Navigation Links */}
          <div className="hidden lg:flex items-center space-x-1">
            {NAV_ITEMS.map((item) => {
              const active = isActive(item.href);
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`px-2.5 py-1 text-[13px] font-sans transition-colors rounded-[3px] ${
                    active
                      ? "text-[var(--foreground)] font-medium bg-[var(--surface-subtle)]"
                      : "text-[var(--foreground-secondary)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)]"
                  }`}
                >
                  {item.name}
                </Link>
              );
            })}
          </div>
        </div>

        {/* Right Shell Controls: Status badge + Theme Toggle */}
        <div className="hidden lg:flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                apiOnline === true
                  ? "bg-[var(--status-ok)]"
                  : apiOnline === false
                  ? "bg-[var(--status-error)]"
                  : "bg-[var(--muted-light)]"
              }`}
            />
            <span className="text-xs font-sans text-[var(--muted)]">
              {apiOnline === true ? "Reference archive" : apiOnline === false ? "Offline" : "Connecting"}
            </span>
          </div>

          <button
            onClick={toggleTheme}
            className="p-1.5 rounded-[3px] text-[var(--foreground-secondary)] hover:text-[var(--foreground)] hover:bg-[var(--surface-subtle)] transition-colors"
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            aria-label="Toggle visual theme"
          >
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          </button>
        </div>

        {/* Mobile menu trigger */}
        <div className="flex lg:hidden items-center gap-2">
          <button
            onClick={toggleTheme}
            className="p-1.5 rounded-[3px] text-[var(--foreground-secondary)] hover:text-[var(--foreground)]"
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="p-1.5 text-[var(--foreground-secondary)] hover:text-[var(--foreground)]"
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </div>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="lg:hidden border-t border-[var(--border)] bg-[var(--surface)] px-4 py-3 space-y-1">
          {NAV_ITEMS.map((item) => {
            const active = isActive(item.href);
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`block px-3 py-2 text-sm font-sans rounded-[3px] ${
                  active
                    ? "text-[var(--foreground)] font-medium bg-[var(--surface-subtle)]"
                    : "text-[var(--foreground-secondary)] hover:text-[var(--foreground)] hover:bg-[var(--surface-hover)]"
                }`}
              >
                {item.name}
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
}

export default Navbar;
