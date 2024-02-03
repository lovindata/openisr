import { ReactNode } from "react";

interface Props {
  children: ReactNode;
  dashed?: boolean;
  roundedFull?: boolean;
  className?: string;
  onClick?: React.MouseEventHandler<HTMLDivElement>;
}

export function BorderBox({
  children,
  dashed,
  roundedFull,
  className,
  onClick,
}: Props) {
  return (
    <div
      className={
        "border" +
        (dashed ? " border-dashed" : "") +
        (roundedFull ? " rounded-full" : " rounded-lg") +
        (className ? ` ${className}` : "")
      }
      onClick={onClick}
    >
      {children}
    </div>
  );
}
