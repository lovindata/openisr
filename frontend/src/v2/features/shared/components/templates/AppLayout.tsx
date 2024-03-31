import NavBar from "@/v2/features/shared/components/organisms/NavBar";
import { ModalProvider } from "@/v2/features/shared/hooks/contexts/Modal";
import { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export function AppLayout({ children }: Props) {
  return (
    <ModalProvider>
      <div className="md:flex md:flex-row">
        <NavBar />
        <div className="h-full w-full">{children}</div>
      </div>
    </ModalProvider>
  );
}
