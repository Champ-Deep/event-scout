import React from 'react';

export const Button: React.FC<
  React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: 'primary' | 'secondary' | 'danger' }
> = ({
  variant = 'primary',
  className = '',
  children,
  ...props
}) => {
  const base = 'px-4 py-2 rounded-lg font-medium text-sm transition-colors disabled:opacity-50 cursor-pointer';
  const styles =
    variant === 'primary'
      ? 'bg-emerald-600 hover:bg-emerald-500 text-white font-semibold shadow-xs'
      : variant === 'danger'
      ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-xs'
      : 'bg-card hover:bg-secondary text-body border border-border shadow-xs';

  return (
    <button className={`${base} ${styles} ${className}`} {...props}>
      {children}
    </button>
  );
};
